# Owner(s): ["module: distributions"]

import torch
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestUtils(TestCase):
    @parametrize(
        "shape",
        [
            (2, 2),
            (3, 3),
            (2, 4, 4),
            (2, 2, 4, 4),
        ],
    )
    def test_tril_matrix_to_vec(self, device, shape):
        mat = torch.randn(shape, device=device)
        n = mat.shape[-1]
        for diag in range(-n, n):
            actual = mat.tril(diag)
            vec = tril_matrix_to_vec(actual, diag)
            tril_mat = vec_to_tril_matrix(vec, diag)
            self.assertEqual(tril_mat, actual, msg=f"diag={diag}, shape={shape}")


instantiate_device_type_tests(TestUtils, globals())


if __name__ == "__main__":
    run_tests()
    
