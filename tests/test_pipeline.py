"""Synthetic, CPU-only checks; no competition data, checkpoints, or downloads."""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from rgb_spatial_refiner import RGBRefiner
from si_reclassifier import extract, redistribute, targets
from tile_d4_inference import TRANSFORMS, apply_t, invert_t, predict_d4
from training.common import check_config, dump, load_probs, sha
from training.data import COLORS, ious, read_pairs, resize_pair

torch.set_num_threads(2)


class NumericalContracts(unittest.TestCase):
    def test_absent_class_is_union_based(self):
        gt = np.array([[1, 1], [2, 3]], dtype=np.uint8)
        correct = ious(gt, gt, 'skip')
        self.assertTrue(np.isnan(correct[0]))
        self.assertEqual(float(np.nanmean(correct)), 1.)
        self.assertEqual(float(ious(gt, gt, '0').mean()), .75)
        self.assertEqual(float(ious(gt, gt, '1').mean()), 1.)
        pred = gt.copy(); pred[0, 0] = 0
        for policy in ('skip', '0', '1'):
            self.assertEqual(ious(pred, gt, policy)[0], 0.)
        self.assertAlmostEqual(float(np.nanmean(ious(pred, gt, 'skip'))), .625)

    def test_d4_inverse_and_native_restore_commute(self):
        x = torch.arange(4*20*20).reshape(1, 4, 20, 20).float()
        variants = []
        for k, flip in TRANSFORMS:
            transformed = apply_t(x, k, flip)
            self.assertTrue(torch.equal(invert_t(transformed, k, flip), x))
            self.assertTrue(torch.equal(F.avg_pool2d(x, 2), invert_t(F.avg_pool2d(transformed, 2), k, flip)))
            variants.append(transformed)
        self.assertTrue(all(not torch.equal(variants[i], variants[j]) for i in range(8) for j in range(i)))

    def test_overlap_reconstruction_on_nonsquare_image(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([0., 1., 2., 3.]).view(1, 4, 1, 1).expand(x.shape[0], 4, *x.shape[-2:])
        image = np.zeros((25, 37, 3), dtype=np.uint8)
        maps, cost = predict_d4(Constant().eval(), image, device='cpu', size=16)
        expected = torch.tensor([0., 1., 2., 3.]).softmax(0).numpy()
        for prob in maps.values():
            self.assertEqual(prob.shape, (4, 25, 37))
            np.testing.assert_allclose(prob, np.broadcast_to(expected[:, None, None], prob.shape), atol=1e-6)
        self.assertEqual(cost['forwards'], 8*cost['tiles'])

    def test_zero_initialized_refiner_preserves_safe_probabilities(self):
        torch.manual_seed(7)
        p = torch.rand(2, 4, 17, 23)+.01; p /= p.sum(1, keepdim=True)
        rgb = torch.randn(2, 3, 17, 23)
        model = RGBRefiner().eval()
        with torch.no_grad():
            q = model(rgb, p, False).softmax(1)
        torch.testing.assert_close(q, p, rtol=1e-6, atol=2e-7)
        with torch.no_grad():
            model.residual.weight.fill_(.001)
            first = model(rgb, p, False)
            second = model(rgb*3+4, p, False)
        self.assertTrue(torch.equal(first, second))

    def test_component_redistribution_preserves_non_si_and_mass(self):
        p = np.empty((4, 6, 7), np.float32)
        p[:] = np.array([.05, .1, .05, .8])[:, None, None]
        p[:, 1:3, 1:3] = np.array([.6, .1, .2, .1])[:, None, None]
        p[:, 4:6, 5:7] = np.array([.2, .1, .6, .1])[:, None, None]
        lab, X, _, _ = extract(p)
        self.assertEqual(int(lab.max()), 2)
        self.assertEqual(X[:, :6].shape, (2, 6))
        q = redistribute(p, lab, np.array([0., 1.]))
        np.testing.assert_array_equal(q[[1, 3]], p[[1, 3]])
        np.testing.assert_array_equal(q[:, lab == 0], p[:, lab == 0])
        np.testing.assert_allclose(q[0]+q[2], p[0]+p[2], atol=2e-7)
        self.assertTrue((q.argmax(0)[lab == 1] == 2).all())
        self.assertTrue((q.argmax(0)[lab == 2] == 0).all())
        target = np.ones((6, 7), np.uint8); target[lab == 1] = 0
        labels, _ = targets(lab, target)
        np.testing.assert_array_equal(labels, [1, -1])

    def test_rgb_labels_resize_as_categories(self):
        gt = np.array([[0, 1], [2, 3]], np.uint8)
        image = np.repeat((gt*70)[:, :, None], 3, axis=2)
        _, enlarged = resize_pair(image, gt, 4)
        np.testing.assert_array_equal(enlarged, np.repeat(np.repeat(gt, 2, 0), 2, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for name, pixels in [('images/example_image.png', image), ('masks/example_mask.png', COLORS[gt])]:
                    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
                    archive.writestr('train/'+name, stream.getvalue())
            pairs = read_pairs(path)
            self.assertEqual(pairs[0][0], 'example')
            np.testing.assert_array_equal(pairs[0][2], gt)

    def test_cache_rejects_wrong_contract_and_modified_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'example').mkdir()
            path = root/'example/prob.npy'; np.save(path, np.full((4, 2, 3), .25, np.float32))
            dump(root/'manifest.json', {'fold': 0, 'probability_sha256': {'example': sha(path)}})
            self.assertEqual(load_probs(root, ['example'], {'fold': 0})['example'].shape, (4, 2, 3))
            with self.assertRaises(ValueError):
                load_probs(root, ['example'], {'fold': 1})
            np.save(path, np.zeros((4, 2, 3), np.float32))
            with self.assertRaises(ValueError):
                load_probs(root, ['example'], {'fold': 0})

    def test_recorded_folds_preserve_group_isolation(self):
        manifest = json.loads((ROOT/'resources/folds.json').read_text(encoding='utf-8'))
        rows = manifest['images']
        self.assertEqual(len(rows), 70)
        self.assertEqual(len({row['id'] for row in rows}), 70)
        for fold in range(5):
            held = [r for r in rows if r['fold'] == fold]
            train = [r for r in rows if r['fold'] != fold]
            self.assertEqual(len(held), 14)
            self.assertFalse({r['group'] for r in held} & {r['group'] for r in train})

    def test_wrong_fold_checkpoint_is_rejected(self):
        check_config({'config': {'fold': 1}}, {'fold': 1})
        with self.assertRaises(ValueError):
            check_config({'config': {'fold': 0}}, {'fold': 1})
        with self.assertRaises(ValueError):
            check_config({}, {'fold': 1})


if __name__ == '__main__':
    unittest.main()
