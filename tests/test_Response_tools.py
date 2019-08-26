import numpy as np
from Response_tools import Response_tools
import pylibxc
from gbasis.evals.density import evaluate_density
from gbasis.evals.eval import evaluate_basis
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeTF
from grid.utils import get_cov_radii
from grid.atomic_grid import AtomicGrid
from grid.molgrid import MolGrid
from iodata import load_one
from gbasis.wrappers import from_iodata
import pytest

molecule = load_one("h2o.fchk")
basis = from_iodata(molecule)
res_hf = Response_tools(basis, molecule, coord_type="cartesian", k="HF")
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
res_lda = Response_tools(
    basis, molecule, coord_type="cartesian", k="lda_x", molgrid=molgrid
)
two_electron_int = electron_repulsion_integral(
    basis, transform=molecule.mo.coeffs.T, coord_type="cartesian", notation="chemist"
)


def test_Response_tools_Frontier_MO_index():
    assert res_hf.Frontier_MO_index(sign="plus", spin="alpha") == [[5], []]
    assert res_hf.Frontier_MO_index(sign="minus", spin="alpha") == [[4], []]
    assert res_hf.Frontier_MO_index(sign="plus", spin="beta") == [[], [24]]
    assert res_hf.Frontier_MO_index(sign="minus", spin="beta") == [[], [23]]
    assert res_lda.Frontier_MO_index(sign="plus", spin="alpha") == [[5], []]
    assert res_lda.Frontier_MO_index(sign="minus", spin="alpha") == [[4], []]
    assert res_lda.Frontier_MO_index(sign="plus", spin="beta") == [[], [24]]
    assert res_lda.Frontier_MO_index(sign="minus", spin="beta") == [[], [23]]


def test_Respons_tools_mhu():
    assert np.allclose(res_hf.mhu(sign="plus", spin="alpha"), 0.08054455)
    assert np.allclose(res_hf.mhu(sign="minus", spin="alpha"), -0.30456815)
    assert np.allclose(res_hf.mhu(sign="plus", spin="beta"), 0.08054455)
    assert np.allclose(res_hf.mhu(sign="minus", spin="beta"), -0.30456815)
    assert np.allclose(res_lda.mhu(sign="plus", spin="alpha"), 0.08054455)
    assert np.allclose(res_lda.mhu(sign="minus", spin="alpha"), -0.30456815)
    assert np.allclose(res_lda.mhu(sign="plus", spin="beta"), 0.08054455)
    assert np.allclose(res_lda.mhu(sign="minus", spin="beta"), -0.30456815)


def test_Response_tools_eta():
    assert np.allclose(
        res_hf.eta(sign=("plus", "plus"), spin=("alpha", "alpha")),
        np.array([[-0.05092241]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "plus"), spin=("alpha", "beta")),
        np.array([[0.39184659]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "plus"), spin=("beta", "alpha")),
        np.array([[0.39184659]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "plus"), spin=("beta", "beta")),
        np.array([[-0.05092241]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "minus"), spin=("alpha", "alpha")),
        np.array([[-0.23233698]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "minus"), spin=("alpha", "beta")),
        np.array([[0.59229919]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "minus"), spin=("beta", "alpha")),
        np.array([[0.59229919]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "minus"), spin=("beta", "beta")),
        np.array([[-0.23233698]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "plus"), spin=("alpha", "alpha")),
        np.array([[-0.23233698]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "plus"), spin=("alpha", "beta")),
        np.array([[0.59229919]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "plus"), spin=("beta", "alpha")),
        np.array([[0.59229919]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "plus"), spin=("beta", "beta")),
        np.array([[-0.23233698]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "minus"), spin=("alpha", "alpha")),
        np.array([[-1.21767093]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "minus"), spin=("alpha", "beta")),
        np.array([[1.84057396]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "minus"), spin=("beta", "alpha")),
        np.array([[1.84057396]]),
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "minus"), spin=("beta", "beta")),
        np.array([[-1.21767093]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("alpha", "alpha")),
        np.array([[0.17084965]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("alpha", "beta")),
        np.array([[0.33201827]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("beta", "alpha")),
        np.array([[0.33201827]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("beta", "beta")),
        np.array([[0.17084965]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("alpha", "alpha")),
        np.array([[0.32868757]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("alpha", "beta")),
        np.array([[0.33654998]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("beta", "alpha")),
        np.array([[0.33654998]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("beta", "beta")),
        np.array([[0.32868757]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("alpha", "alpha")),
        np.array([[0.32868757]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("alpha", "beta")),
        np.array([[0.33654998]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("beta", "alpha")),
        np.array([[0.33654998]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("beta", "beta")),
        np.array([[0.32868757]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("alpha", "alpha")),
        np.array([[0.48697128]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("alpha", "beta")),
        np.array([[0.57420769]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("beta", "alpha")),
        np.array([[0.57420769]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("beta", "beta")),
        np.array([[0.48697128]]),
    )


def test_Response_tools_fukui():
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
    M_HF = np.zeros([140, 140], float)
    M_LDA = np.zeros([140, 140], float)
    K_lla_lda = np.zeros([1, 140], float)
    K_hha_lda = np.zeros([1, 140], float)
    K_llb_lda = np.zeros([1, 140], float)
    K_hhb_lda = np.zeros([1, 140], float)
    K_lla_hf = np.zeros([1, 140], float)
    K_hha_hf = np.zeros([1, 140], float)
    K_llb_hf = np.zeros([1, 140], float)
    K_hhb_hf = np.zeros([1, 140], float)
    for s in range(2):
        for i_c, i in enumerate(indices[0][s]):
            for a_c, a in enumerate(indices[1][s]):
                ias = (
                    s * len(indices[0][0]) * len(indices[1][0])
                    + i_c * len(indices[1][s])
                    + a_c
                )
                K_lla_lda[0][ias] = K_lla_lda[0][ias] = two_electron_int[
                    5, 5, i, a
                ] + molgrid.integrate(
                    f_xc_values[s]
                    * MO_values_conjugated[5]
                    * MO_values[5]
                    * MO_values_conjugated[i]
                    * MO_values[a]
                )
                K_hha_lda[0][ias] = K_hha_lda[0][ias] = two_electron_int[
                    4, 4, i, a
                ] + molgrid.integrate(
                    f_xc_values[s]
                    * MO_values_conjugated[4]
                    * MO_values[4]
                    * MO_values_conjugated[i]
                    * MO_values[a]
                )
                K_llb_lda[0][ias] = K_llb_lda[0][ias] = two_electron_int[
                    24, 24, i, a
                ] + molgrid.integrate(
                    f_xc_values[s + 1]
                    * MO_values_conjugated[24]
                    * MO_values[24]
                    * MO_values_conjugated[i]
                    * MO_values[a]
                )
                K_hhb_lda[0][ias] = K_hhb_lda[0][ias] = two_electron_int[
                    23, 23, i, a
                ] + molgrid.integrate(
                    f_xc_values[s + 1]
                    * MO_values_conjugated[23]
                    * MO_values[23]
                    * MO_values_conjugated[i]
                    * MO_values[a]
                )
                K_lla_hf[0][ias] = K_lla_hf[0][ias] + two_electron_int[5, 5, i, a]
                K_hha_hf[0][ias] = K_hha_hf[0][ias] + two_electron_int[4, 4, i, a]
                K_llb_hf[0][ias] = K_llb_hf[0][ias] + two_electron_int[24, 24, i, a]
                K_hhb_hf[0][ias] = K_hhb_hf[0][ias] + two_electron_int[23, 23, i, a]
                if s == 0:
                    K_lla_hf[0][ias] = K_lla_hf[0][ias] - two_electron_int[5, 5, i, a]
                    K_hha_hf[0][ias] = K_hha_hf[0][ias] - two_electron_int[4, 4, i, a]
                if s == 1:
                    K_llb_hf[0][ias] = K_llb_hf[0][ias] - two_electron_int[24, 24, i, a]
                    K_hhb_hf[0][ias] = K_hhb_hf[0][ias] - two_electron_int[23, 23, i, a]
                for t in range(2):
                    for j_c, j in enumerate(indices[0][t]):
                        for b_c, b in enumerate(indices[1][t]):
                            jbt = (
                                t * len(indices[0][0]) * len(indices[1][0])
                                + j_c * len(indices[1][t])
                                + b_c
                            )
                            M_HF[ias][jbt] = (
                                M_HF[ias][jbt] + 2 * two_electron_int[i, a, j, b]
                            )
                            M_LDA[ias][jbt] = (
                                M_LDA[ias][jbt] + 2 * two_electron_int[i, a, j, b]
                            )
                            values = (
                                f_xc_values[s + t]
                                * MO_values_conjugated[i]
                                * MO_values[a]
                                * MO_values_conjugated[j]
                                * MO_values[b]
                            )
                            M_LDA[ias][jbt] = M_LDA[ias][jbt] + 2 * molgrid.integrate(
                                values
                            )
                            if s == t:
                                M_HF[ias][jbt] = (
                                    M_HF[ias][jbt] - 2 * two_electron_int[i, a, j, b]
                                )
                            if s == t and i == j and a == b:
                                M_HF[ias][jbt] = (
                                    M_HF[ias][jbt]
                                    + molecule.mo.energies[a]
                                    - molecule.mo.energies[i]
                                )
                                M_LDA[ias][jbt] = (
                                    M_LDA[ias][jbt]
                                    + molecule.mo.energies[a]
                                    - molecule.mo.energies[i]
                                )
    M_HF = np.linalg.inv(M_HF)
    M_LDA = np.linalg.inv(M_LDA)
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_lla_lda + K_lla_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_lla_lda + K_lla_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_lla_lda + K_lla_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_lla_lda + K_lla_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_a_lda = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hha_lda + K_hha_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hha_lda + K_hha_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hha_lda + K_hha_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hha_lda + K_hha_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_a_lda = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_llb_lda + K_llb_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_llb_lda + K_llb_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_llb_lda + K_llb_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_llb_lda + K_llb_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_b_lda = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hhb_lda + K_hhb_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hhb_lda + K_hhb_lda, M_LDA)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hhb_lda + K_hhb_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hhb_lda + K_hhb_lda, M_LDA)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_b_lda = np.array([a, b])

    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_lla_hf + K_lla_hf, M_HF)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_lla_hf + K_lla_hf, M_HF)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_lla_hf + K_lla_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_lla_hf + K_lla_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_a_hf = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hha_hf + K_hha_hf, M_HF)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hha_hf + K_hha_hf, M_HF)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hha_hf + K_hha_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hha_hf + K_hha_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_a_hf = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_llb_hf + K_llb_hf, M_HF)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_llb_hf + K_llb_hf, M_HF)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_llb_hf + K_llb_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_llb_hf + K_llb_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_b_hf = np.array([a, b])
    a = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hhb_hf + K_hhb_hf, M_HF)[:, :70]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hhb_hf + K_hhb_hf, M_HF)[:, :70]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_hhb_hf + K_hhb_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_hhb_hf + K_hhb_hf, M_HF)[:, 70:]).reshape(
                    [len(indices[1][0]), len(indices[0][0])]
                ),
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_b_hf = np.array([a, b])
    fukui_p_a_lda[0, 5, 5] = 1
    fukui_m_a_lda[0, 4, 4] = 1
    fukui_p_b_lda[1, 5, 5] = 1
    fukui_m_b_lda[1, 4, 4] = 1
    fukui_p_a_hf[0, 5, 5] = 1
    fukui_m_a_hf[0, 4, 4] = 1
    fukui_p_b_hf[1, 5, 5] = 1
    fukui_m_b_hf[1, 4, 4] = 1
    assert np.allclose(
        res_hf.fukui(sign="plus", spin=("alpha", "alpha")), fukui_p_a_hf[0]
    )
    assert np.allclose(
        res_hf.fukui(sign="plus", spin=("alpha", "beta")), fukui_p_a_hf[1]
    )
    assert np.allclose(
        res_hf.fukui(sign="plus", spin=("beta", "alpha")), fukui_p_b_hf[0]
    )
    assert np.allclose(
        res_hf.fukui(sign="plus", spin=("beta", "beta")), fukui_p_b_hf[1]
    )
    assert np.allclose(
        res_hf.fukui(sign="minus", spin=("alpha", "alpha")), fukui_m_a_hf[0]
    )
    assert np.allclose(
        res_hf.fukui(sign="minus", spin=("alpha", "beta")), fukui_m_a_hf[1]
    )
    assert np.allclose(
        res_hf.fukui(sign="minus", spin=("beta", "alpha")), fukui_m_b_hf[0]
    )
    assert np.allclose(
        res_hf.fukui(sign="minus", spin=("beta", "beta")), fukui_m_b_hf[1]
    )
    assert np.allclose(
        res_lda.fukui(sign="plus", spin=("alpha", "alpha")), fukui_p_a_lda[0]
    )
    assert np.allclose(
        res_lda.fukui(sign="plus", spin=("alpha", "beta")), fukui_p_a_lda[1]
    )
    assert np.allclose(
        res_lda.fukui(sign="plus", spin=("beta", "alpha")), fukui_p_b_lda[0]
    )
    assert np.allclose(
        res_lda.fukui(sign="plus", spin=("beta", "beta")), fukui_p_b_lda[1]
    )
    assert np.allclose(
        res_lda.fukui(sign="minus", spin=("alpha", "alpha")), fukui_m_a_lda[0]
    )
    assert np.allclose(
        res_lda.fukui(sign="minus", spin=("alpha", "beta")), fukui_m_a_lda[1]
    )
    assert np.allclose(
        res_lda.fukui(sign="minus", spin=("beta", "alpha")), fukui_m_b_lda[0]
    )
    assert np.allclose(
        res_lda.fukui(sign="minus", spin=("beta", "beta")), fukui_m_b_lda[1]
    )


def test_Response_tools_linear_response():

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
    M_HF = np.zeros([140, 140], float)
    M_LDA = np.zeros([140, 140], float)
    for s in range(2):
        for i_c, i in enumerate(indices[0][s]):
            for a_c, a in enumerate(indices[1][s]):
                ias = (
                    s * len(indices[0][0]) * len(indices[1][0])
                    + i_c * len(indices[1][s])
                    + a_c
                )
                for t in range(2):
                    for j_c, j in enumerate(indices[0][t]):
                        for b_c, b in enumerate(indices[1][t]):
                            jbt = (
                                t * len(indices[0][0]) * len(indices[1][0])
                                + j_c * len(indices[1][t])
                                + b_c
                            )
                            M_HF[ias][jbt] = (
                                M_HF[ias][jbt] + 2 * two_electron_int[i, a, j, b]
                            )
                            M_LDA[ias][jbt] = (
                                M_LDA[ias][jbt] + 2 * two_electron_int[i, a, j, b]
                            )
                            values = (
                                f_xc_values[s + t]
                                * MO_values_conjugated[i]
                                * MO_values[a]
                                * MO_values_conjugated[j]
                                * MO_values[b]
                            )
                            M_LDA[ias][jbt] = M_LDA[ias][jbt] + 2 * molgrid.integrate(
                                values
                            )
                            if s == t:
                                M_HF[ias][jbt] = (
                                    M_HF[ias][jbt] - 2 * two_electron_int[i, a, j, b]
                                )
                            if s == t and i == j and a == b:
                                M_HF[ias][jbt] = (
                                    M_HF[ias][jbt]
                                    + molecule.mo.energies[a]
                                    - molecule.mo.energies[i]
                                )
                                M_LDA[ias][jbt] = (
                                    M_LDA[ias][jbt]
                                    + molecule.mo.energies[a]
                                    - molecule.mo.energies[i]
                                )
    M_HF = -2 * np.linalg.inv(M_HF)
    M_LDA = -2 * np.linalg.inv(M_LDA)
    assert np.allclose(
        res_hf.linear_response(spin=("alpha", "alpha")), M_HF[:70][:, :70]
    )
    assert np.allclose(
        res_hf.linear_response(spin=("alpha", "beta")), M_HF[:70][:, 70:]
    )
    assert np.allclose(
        res_hf.linear_response(spin=("beta", "alpha")), M_HF[70:][:, :70]
    )
    assert np.allclose(res_hf.linear_response(spin=("beta", "beta")), M_HF[70:][:, 70:])
    assert np.allclose(
        res_lda.linear_response(spin=("alpha", "alpha")), M_LDA[:70][:, :70]
    )
    assert np.allclose(
        res_lda.linear_response(spin=("alpha", "beta")), M_LDA[:70][:, 70:]
    )
    assert np.allclose(
        res_lda.linear_response(spin=("beta", "alpha")), M_LDA[70:][:, :70]
    )
    assert np.allclose(
        res_lda.linear_response(spin=("beta", "beta")), M_LDA[70:][:, 70:]
    )


def test_Response_tools_Frontier_MO_index_raise():
    with pytest.raises(TypeError) as error:
        fi = res_lda.Frontier_MO_index(sign="plus", spin=1)
    assert str(error.value) == """'spin' must be 'alpha' or 'beta'"""
    with pytest.raises(TypeError) as error:
        fi = res_lda.Frontier_MO_index(sign=1, spin="alpha")
    assert str(error.value) == """'sign' must be 'plus' or 'minus'"""
    with pytest.raises(ValueError) as error:
        fi = res_lda.Frontier_MO_index(sign="plus", spin="abcd")
    assert str(error.value) == """'spin' must be 'alpha' or 'beta'"""
    with pytest.raises(ValueError) as error:
        fi = res_lda.Frontier_MO_index(sign="abcd", spin="alpha")
    assert str(error.value) == """'sign' must be 'plus' or 'minus'"""


def test_Response_tools_eta_raise():
    with pytest.raises(TypeError) as error:
        et = res_lda.eta(sign=("plus", "plus"), spin=1)
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.eta(sign=("plus", "plus"), spin=("a", "b", "c"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.eta(sign=("plus", "plus"), spin=("alpha", "b"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.eta(sign=("plus", "plus"), spin=("a", "beta"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(TypeError) as error:
        fi = res_lda.eta(sign=1, spin=("alpha", "alpha"))
    assert (
        str(error.value)
        == """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
    )
    with pytest.raises(ValueError) as error:
        fi = res_lda.eta(sign=("plus", "plus", "plus"), spin=("alpha", "alpha"))
    assert (
        str(error.value)
        == """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
    )
    with pytest.raises(ValueError) as error:
        fi = res_lda.eta(sign=("abcd", "plus"), spin=("alpha", "alpha"))
    assert (
        str(error.value)
        == """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
    )
    with pytest.raises(ValueError) as error:
        fi = res_lda.eta(sign=("plus", "abcd"), spin=("alpha", "alpha"))
    assert (
        str(error.value)
        == """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
    )


def test_Response_tools_fukui_raise():
    with pytest.raises(TypeError) as error:
        et = res_lda.fukui(sign="plus", spin=1)
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.fukui(sign="plus", spin=("a", "b", "c"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.fukui(sign="plus", spin=("alpha", "b"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.fukui(sign="plus", spin=("a", "beta"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(TypeError) as error:
        et = res_lda.fukui(sign=1, spin=("alpha", "beta"))
    assert str(error.value) == """'sign' must be 'plus' or 'minus'"""
    with pytest.raises(ValueError) as error:
        et = res_lda.fukui(sign="abcd", spin=("alpha", "beta"))
    assert str(error.value) == """'sign' must be 'plus' or 'minus'"""


def test_Response_tools_linear_response_raise():
    with pytest.raises(TypeError) as error:
        et = res_lda.linear_response(spin=1)
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.linear_response(spin=("a", "b", "c"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.linear_response(spin=("alpha", "b"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.linear_response(spin=("a", "beta"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
