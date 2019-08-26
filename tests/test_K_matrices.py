import numpy as np
from gbasis.evals.density import evaluate_density
from gbasis.evals.eval import evaluate_basis
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
import pylibxc
from K_matrices import K_matrices
from iodata import load_one
from gbasis.wrappers import from_iodata
from numpy.testing import assert_allclose
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeTF
from grid.utils import get_cov_radii
from grid.atomic_grid import AtomicGrid
from grid.molgrid import MolGrid
import pytest
from iodata import IOData

molecule = load_one("h2o.fchk")
basis = from_iodata(molecule)
K_mats = K_matrices(basis, molecule, coord_type="cartesian")
two_electron_int = electron_repulsion_integral(
    basis, transform=molecule.mo.coeffs.T, coord_type="cartesian", notation="chemist"
)
r0 = 1e-10
radii = get_cov_radii(molecule.atnums)
n_rad_points = 100
deg = 5
onedgrid = GaussChebyshev(n_rad_points)
molecule_at_grid = []
for i in range(len(molecule.atnums)):
    R = BeckeTF.find_parameter(onedgrid.points, r0, radii[i])
    rad_grid = BeckeTF(r0, R).transform_grid(onedgrid)
    molecule_at_grid = molecule_at_grid + [
        AtomicGrid.special_init(rad_grid, radii[i], degs=[deg], scales=[])
    ]
molgrid = MolGrid(molecule_at_grid, molecule.atnums)


def test_K_matrices_K_indices():

    occupied_ind = [[], []]
    virtual_ind = [[], []]
    for i in range(len(molecule.mo.occsa)):
        if molecule.mo.occsa[i] - 1 == 0:
            occupied_ind[0] = occupied_ind[0] + [i]
        if molecule.mo.occsa[i] == 0:
            virtual_ind[0] = virtual_ind[0] + [i]
    for i in range(len(molecule.mo.occsb)):
        if molecule.mo.occsb[i] - 1 == 0:
            occupied_ind[1] = occupied_ind[1] + [i + molecule.mo.nbasis]
        if molecule.mo.occsb[i] == 0:
            virtual_ind[1] = virtual_ind[1] + [i + molecule.mo.nbasis]
    shape = "square"
    index = None
    assert K_mats.K_indices(shape=shape, index=index) == (occupied_ind, virtual_ind)
    shape = "line"
    index = [[4], []]
    assert K_mats.K_indices(shape=shape, index=index) == (
        occupied_ind,
        virtual_ind,
        index,
    )
    shape = "point"
    index = [[4], [23]]
    assert K_mats.K_indices(shape=shape, index=index) == (
        occupied_ind,
        virtual_ind,
        index,
    )


def test_K_matrices_K_shape():

    occupied_ind = [[], []]
    virtual_ind = [[], []]
    for i in range(len(molecule.mo.occsa)):
        if molecule.mo.occsa[i] - 1 == 0:
            occupied_ind[0] = occupied_ind[0] + [i]
        if molecule.mo.occsa[i] == 0:
            virtual_ind[0] = virtual_ind[0] + [i]
    for i in range(len(molecule.mo.occsb)):
        if molecule.mo.occsb[i] - 1 == 0:
            occupied_ind[1] = occupied_ind[1] + [i + molecule.mo.nbasis]
        if molecule.mo.occsb[i] == 0:
            virtual_ind[1] = virtual_ind[1] + [i + molecule.mo.nbasis]
    shape = "square"
    index = None
    K_size = len(occupied_ind[0]) * len(virtual_ind[0]) + len(occupied_ind[1]) * len(
        virtual_ind[1]
    )
    assert_allclose(K_mats.K_shape(shape=shape, index=index), [K_size, K_size])
    shape = "line"
    index = [[4], []]
    assert_allclose(K_mats.K_shape(shape=shape, index=index), [1, K_size])
    shape = "point"
    index = [[4, 4], []]
    assert np.allclose(K_mats.K_shape(shape=shape, index=index), [1, 1])


def test_K_matrices_K_coulomb():

    occupied_ind = [[], []]
    virtual_ind = [[], []]
    for i in range(len(molecule.mo.occsa)):
        if molecule.mo.occsa[i] - 1 == 0:
            occupied_ind[0] = occupied_ind[0] + [i]
        if molecule.mo.occsa[i] == 0:
            virtual_ind[0] = virtual_ind[0] + [i]
    for i in range(len(molecule.mo.occsb)):
        if molecule.mo.occsb[i] - 1 == 0:
            occupied_ind[1] = occupied_ind[1] + [i + molecule.mo.nbasis]
        if molecule.mo.occsb[i] == 0:
            virtual_ind[1] = virtual_ind[1] + [i + molecule.mo.nbasis]
    indices = (occupied_ind, virtual_ind)
    K_size = len(occupied_ind[0]) * len(virtual_ind[0]) + len(occupied_ind[1]) * len(
        virtual_ind[1]
    )
    K_shape = [K_size, K_size]
    K_coulomb = np.zeros(K_shape, float)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for i_c, i in enumerate(indices[0][s]):
                        for a_c, a in enumerate(indices[1][s]):
                            ias = (
                                s * len(indices[0][0]) * len(indices[1][0])
                                + i_c * len(indices[1][s])
                                + a_c
                            )
                            K_coulomb[ias][jbt] = (
                                K_coulomb[ias][jbt] + two_electron_int[i, a, j, b]
                            )
    assert np.allclose(K_mats.K_coulomb(), K_coulomb)
    K_shape = [1, K_size]
    K_coulomb = np.zeros(K_shape, float)
    index = [[4], []]
    indices = (occupied_ind, virtual_ind, index)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for f_c, f in enumerate(indices[2][s]):
                        ffs = s * len(indices[2][0]) + f_c
                        K_coulomb[ffs][jbt] = (
                            K_coulomb[ffs][jbt] + two_electron_int[f, f, j, b]
                        )
    assert np.allclose(K_mats.K_coulomb(shape="line", index=index), K_coulomb)
    l = [4, 24]
    K_coulomb = two_electron_int[l[0], l[0], l[1], l[1]]
    assert np.allclose(K_mats.K_coulomb(shape="point", index=[[4], [24]]), K_coulomb)


def test_K_matrices_K_fxc_HF():

    occupied_ind = [[], []]
    virtual_ind = [[], []]
    for i in range(len(molecule.mo.occsa)):
        if molecule.mo.occsa[i] - 1 == 0:
            occupied_ind[0] = occupied_ind[0] + [i]
        if molecule.mo.occsa[i] == 0:
            virtual_ind[0] = virtual_ind[0] + [i]
    for i in range(len(molecule.mo.occsb)):
        if molecule.mo.occsb[i] - 1 == 0:
            occupied_ind[1] = occupied_ind[1] + [i + molecule.mo.nbasis]
        if molecule.mo.occsb[i] == 0:
            virtual_ind[1] = virtual_ind[1] + [i + molecule.mo.nbasis]
    indices = (occupied_ind, virtual_ind)
    K_size = len(occupied_ind[0]) * len(virtual_ind[0]) + len(occupied_ind[1]) * len(
        virtual_ind[1]
    )
    K_shape = [K_size, K_size]
    K_fxc_HF = np.zeros(K_shape, float)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for i_c, i in enumerate(indices[0][s]):
                        for a_c, a in enumerate(indices[1][s]):
                            ias = (
                                s * len(indices[0][0]) * len(indices[1][0])
                                + i_c * len(indices[1][s])
                                + a_c
                            )
                            if s == t:
                                K_fxc_HF[ias][jbt] = (
                                    K_fxc_HF[ias][jbt] - two_electron_int[i, a, j, b]
                                )
    assert np.allclose(K_mats.K_fxc_HF(), K_fxc_HF)
    K_shape = [1, K_size]
    K_fxc_HF = np.zeros(K_shape, float)
    index = [[4], []]
    indices = (occupied_ind, virtual_ind, index)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for f_c, f in enumerate(indices[2][s]):
                        ffs = s * len(indices[2][0]) + f_c
                        if s == t:
                            K_fxc_HF[ffs][jbt] = (
                                K_fxc_HF[ffs][jbt] - two_electron_int[f, f, j, b]
                            )
    assert np.allclose(K_mats.K_fxc_HF(shape="line", index=index), K_fxc_HF)
    K_shape = [1, 1]
    l = [4, 4]
    K_fxc_HF = -two_electron_int[l[0], l[0], l[1], l[1]]
    assert np.allclose(K_mats.K_fxc_HF(shape="point", index=[[4, 4], []]), K_fxc_HF)


def test_K_matrices_K_fxc_DFT():

    xc_function = pylibxc.LibXCFunctional("lda_x", "polarized")
    a_dm = np.dot(molecule.mo.coeffsa * molecule.mo.occsa, molecule.mo.coeffsa.T.conj())
    b_dm = np.dot(molecule.mo.coeffsb * molecule.mo.occsb, molecule.mo.coeffsb.T.conj())
    a_density_values = evaluate_density(
        a_dm, basis, molgrid.points, coord_type="cartesian"
    )
    b_density_values = evaluate_density(
        b_dm, basis, molgrid.points, coord_type="cartesian"
    )
    inp = {"rho": np.array([a_density_values, b_density_values])}
    inp["rho"] = inp["rho"].transpose().reshape(len(inp["rho"][0]) * 2)
    f_xc_values = (xc_function.compute(inp, None, False, False, True, False))["v2rho2"]
    f_xc_values = np.array(
        [
            f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 0],
            f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 1],
            f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 2],
        ]
    )
    del (a_dm, b_dm, a_density_values, b_density_values, xc_function, inp)

    MO_values = evaluate_basis(
        basis, molgrid.points, transform=molecule.mo.coeffs.T, coord_type="cartesian"
    )
    MO_values_conjugated = np.conjugate(MO_values)

    occupied_ind = [[], []]
    virtual_ind = [[], []]
    for i in range(len(molecule.mo.occsa)):
        if molecule.mo.occsa[i] - 1 == 0:
            occupied_ind[0] = occupied_ind[0] + [i]
        if molecule.mo.occsa[i] == 0:
            virtual_ind[0] = virtual_ind[0] + [i]
    for i in range(len(molecule.mo.occsb)):
        if molecule.mo.occsb[i] - 1 == 0:
            occupied_ind[1] = occupied_ind[1] + [i + molecule.mo.nbasis]
        if molecule.mo.occsb[i] == 0:
            virtual_ind[1] = virtual_ind[1] + [i + molecule.mo.nbasis]
    indices = (occupied_ind, virtual_ind)
    K_size = len(occupied_ind[0]) * len(virtual_ind[0]) + len(occupied_ind[1]) * len(
        virtual_ind[1]
    )
    K_shape = [K_size, K_size]
    K_fxc_DFT = np.zeros(K_shape, float)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for i_c, i in enumerate(indices[0][s]):
                        for a_c, a in enumerate(indices[1][s]):
                            ias = (
                                s * len(indices[0][0]) * len(indices[1][0])
                                + i_c * len(indices[1][s])
                                + a_c
                            )
                            values = (
                                f_xc_values[s + t]
                                * MO_values_conjugated[i]
                                * MO_values[a]
                                * MO_values_conjugated[j]
                                * MO_values[b]
                            )
                            K_fxc_DFT[ias][jbt] = K_fxc_DFT[ias][
                                jbt
                            ] + molgrid.integrate(values)
    assert np.allclose(
        K_mats.K_fxc_DFT(XC_functional="lda_x", molgrid=molgrid), K_fxc_DFT
    )
    K_shape = [1, K_size]
    K_fxc_DFT = np.zeros(K_shape, float)
    index = [[4], []]
    indices = (occupied_ind, virtual_ind, index)
    for t in range(2):
        for j_c, j in enumerate(indices[0][t]):
            for b_c, b in enumerate(indices[1][t]):
                jbt = (
                    t * len(indices[0][0]) * len(indices[1][0])
                    + j_c * len(indices[1][t])
                    + b_c
                )
                for s in range(2):
                    for f_c, f in enumerate(indices[2][s]):
                        ffs = s * len(indices[2][0]) + f_c
                        values = (
                            f_xc_values[s + t]
                            * MO_values_conjugated[f]
                            * MO_values[f]
                            * MO_values_conjugated[j]
                            * MO_values[b]
                        )
                        K_fxc_DFT[ffs][jbt] = K_fxc_DFT[ffs][jbt] + molgrid.integrate(
                            values
                        )
    index = [[4], []]
    assert np.allclose(
        K_mats.K_fxc_DFT(
            shape="line", index=index, XC_functional="lda_x", molgrid=molgrid
        ),
        K_fxc_DFT,
    )
    K_shape = [1, 1]
    values = (
        f_xc_values[0]
        * MO_values_conjugated[4]
        * MO_values[4]
        * MO_values_conjugated[4]
        * MO_values[4]
    )
    K_fxc_DFT = molgrid.integrate(values)
    assert np.allclose(
        K_mats.K_fxc_DFT(
            shape="point", index=[[4, 4], []], XC_functional="lda_x", molgrid=molgrid
        ),
        K_fxc_DFT,
    )


def test_K_matrices_raise_init():
    with pytest.raises(TypeError) as error:
        K_ma = K_matrices(1, molecule, coord_type="cartesian")
    assert (
        str(error.value)
        == """'basis' must be a list of 'GeneralizedContractionShell' instances"""
    )
    with pytest.raises(TypeError) as error:
        K_ma = K_matrices([1], molecule, coord_type="cartesian")
    assert (
        str(error.value)
        == """'basis' must be a list of 'GeneralizedContractionShell' instances"""
    )
    with pytest.raises(TypeError) as error:
        K_ma = K_matrices(basis, 1, coord_type="cartesian")
    assert str(error.value) == """'molecule' must be a 'IOData' instance"""
    a = IOData()
    with pytest.raises(TypeError) as error:
        K_ma = K_matrices(basis, a, coord_type="cartesian")
    assert (
        str(error.value) == """'molecule.mo' must be a 'MolecularOrbitals' instance"""
    )
    with pytest.raises(ValueError) as error:
        K_ma = K_matrices(basis, molecule, coord_type="1")
    assert (
        str(error.value)
        == """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
    )
    with pytest.raises(ValueError) as error:
        K_ma = K_matrices(basis, molecule, coord_type=["1"])
    assert (
        str(error.value)
        == """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
    )
    with pytest.raises(TypeError) as error:
        K_ma = K_matrices(basis, molecule, coord_type=3)
    assert (
        str(error.value)
        == """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
    )


def test_K_matrices_raise_K_indices():
    with pytest.raises(TypeError) as error:
        ind = K_mats.K_indices(shape=2)
    assert str(error.value) == """'shape' must be 'square' or 'line' or 'point' """
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="rectangle")
    assert str(error.value) == """'shape' must be 'square' or 'line' or 'point' """
    with pytest.raises(TypeError) as error:
        ind = K_mats.K_indices(shape="line", index=(1))
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
    )
    with pytest.raises(TypeError) as error:
        ind = K_mats.K_indices(shape="line", index=[[1], [2], [3]])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
    )
    with pytest.raises(TypeError) as error:
        ind = K_mats.K_indices(shape="line", index=[(1), [2]])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
    )
    with pytest.raises(TypeError) as error:
        ind = K_mats.K_indices(shape="line", index=[[1], (2)])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="line", index=[[1], [1]])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="line", index=[[], [2]])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="line", index=[[21], []])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="point", index=[[1], []])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="point", index=[[], [5, 23]])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
    )
    with pytest.raises(ValueError) as error:
        ind = K_mats.K_indices(shape="point", index=[[21, 2], []])
    assert (
        str(error.value)
        == """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
    )


def test_K_matrices_raise_K_coulomb():
    with pytest.raises(TypeError) as error:
        k_coul = K_mats.K_coulomb(conjugate=2)
    assert str(error.value) == """'conjugate' must be a bool"""


def test_K_matrices_raise_K_fxc_HF():
    with pytest.raises(TypeError) as error:
        k_fxc_hf = K_mats.K_fxc_HF(conjugate=2)
    assert str(error.value) == """'conjugate' must be a bool"""


def test_K_matrices_raise_K_fxc_DFT():
    with pytest.raises(TypeError) as error:
        k_fxc_dft = K_mats.K_fxc_DFT(
            XC_functional="lda_x", conjugate=2, molgrid=molgrid
        )
    assert str(error.value) == """'conjugate' must be a bool"""
    with pytest.raises(TypeError) as error:
        k_fxc_dft = K_mats.K_fxc_DFT(XC_functional="lda_x", conjugate=False, molgrid=2)
    assert str(error.value) == """'molgrid' must be a 'MolGrid' instance"""
    with pytest.raises(TypeError) as error:
        k_fxc_dft = K_mats.K_fxc_DFT(XC_functional=2, conjugate=False, molgrid=molgrid)
    assert str(error.value) == """'XC_functional' must be a str"""
    with pytest.raises(ValueError) as error:
        k_fxc_dft = K_mats.K_fxc_DFT(
            XC_functional="abcd", conjugate=False, molgrid=molgrid
        )
    assert (
        str(error.value)
        == """"Not suported functionnal, see pylibxc.util.xc_available_functional_names() or the webpage: https://tddft.org/programs/libxc/functionals/"""
    )
    with pytest.raises(ValueError) as error:
        k_fxc_dft = K_mats.K_fxc_DFT(
            XC_functional="hyb_mgga_xc_pwb6k", conjugate=False, molgrid=molgrid
        )
    assert (
        str(error.value)
        == """"Not suported functionnal, only the LDA ans GGAs are supported"""
    )
