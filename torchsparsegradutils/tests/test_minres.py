# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import unittest

import torch

import torchsparsegradutils
import random
from torchsparsegradutils.utils.minres import minres


class TestMinres(unittest.TestCase):
    def setUp(self):
        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    def _test_minres(self, rhs_shape, shifts=None, matrix_batch_shape=torch.Size([])):
        size = rhs_shape[-2] if len(rhs_shape) > 1 else rhs_shape[-1]
        rhs = torch.randn(rhs_shape, dtype=torch.float64)

        matrix = torch.randn(*matrix_batch_shape, size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.mT)
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(size, dtype=torch.float64).mul_(1e-1))

        # Compute solves with minres
        if shifts is not None:
            shifts = shifts.type_as(rhs)

        settings = torchsparsegradutils.utils.MINRESSettings(minres_tolerance=1e-6)
        solves = minres(matrix, rhs=rhs, value=-1, shifts=shifts, settings=settings)

        # Make sure that we're not getting weird batch dim effects
        while matrix.dim() < len(rhs_shape):
            matrix = matrix.unsqueeze(0)

        # Maybe add shifts
        if shifts is not None:
            matrix = matrix - torch.mul(
                torch.eye(size, dtype=torch.float64), shifts.view(*shifts.shape, *[1 for _ in matrix.shape])
            )

        # Compute solves exactly
        actual = torch.linalg.solve(-matrix, rhs.unsqueeze(-1) if rhs.dim() == 1 else rhs)
        if rhs.dim() == 1:
            actual = actual.squeeze(-1)

        self.assertAllClose(solves, actual, atol=1e-3, rtol=1e-4)

    def test_minres_vec(self):
        return self._test_minres(torch.Size([20]))

    def test_minres_vec_multiple_shifts(self):
        shifts = torch.tensor([0.0, 1.0, 2.0])
        return self._test_minres(torch.Size([5]), shifts=shifts)

    def test_minres_mat(self):
        self._test_minres(torch.Size([20, 5]))
        self._test_minres(torch.Size([3, 20, 5]))
        self._test_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]))
        return self._test_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]))

    def test_minres_mat_multiple_shifts(self):
        shifts = torch.tensor([0.0, 1.0, 2.0])
        self._test_minres(torch.Size([20, 5]), shifts=shifts)
        self._test_minres(torch.Size([3, 20, 5]), shifts=shifts)
        self._test_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)
        return self._test_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)

    def assertAllClose(self, tensor1, tensor2, rtol=1e-4, atol=1e-5, equal_nan=False):
        if not tensor1.shape == tensor2.shape:
            raise ValueError(f"tensor1 ({tensor1.shape}) and tensor2 ({tensor2.shape}) do not have the same shape.")

        if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan):
            return True

        if not equal_nan:
            if not torch.equal(tensor1, tensor1):
                raise AssertionError(f"tensor1 ({tensor1.shape}) contains NaNs")
            if not torch.equal(tensor2, tensor2):
                raise AssertionError(f"tensor2 ({tensor2.shape}) contains NaNs")

        rtol_diff = (torch.abs(tensor1 - tensor2) / torch.abs(tensor2)).view(-1)
        rtol_diff = rtol_diff[torch.isfinite(rtol_diff)]
        rtol_max = rtol_diff.max().item()

        atol_diff = (torch.abs(tensor1 - tensor2) - torch.abs(tensor2).mul(rtol)).view(-1)
        atol_diff = atol_diff[torch.isfinite(atol_diff)]
        atol_max = atol_diff.max().item()

        raise AssertionError(
            f"tensor1 ({tensor1.shape}) and tensor2 ({tensor2.shape}) are not close enough. \n"
            f"max rtol: {rtol_max:0.8f}\t\tmax atol: {atol_max:0.8f}"
        )

    def assertEqual(self, item1, item2):
        if torch.is_tensor(item1) and torch.is_tensor(item2):
            if torch.equal(item1, item2):
                return True
            else:
                raise AssertionError(f"{item1} does not equal {item2}.")
        elif torch.is_tensor(item1) or torch.is_tensor(item2):
            raise AssertionError(f"item1 ({type(item1)}) and item2 ({type(item2)}) are not the same type.")
        elif item1 == item2:
            return True
        elif type(item1) is type(item2):
            raise AssertionError(f"item1 ({type(item1)}) and item2 ({type(item2)}) are not the same type.")
        else:
            raise AssertionError(f"tensor1 ({item1}) does not equal tensor2 ({item2}).")
