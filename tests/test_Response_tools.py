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
        res_hf.eta(sign=("plus", "plus"), spin=("alpha", "alpha")), np.array([[0.0]])
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
        res_hf.eta(sign=("plus", "plus"), spin=("beta", "beta")), np.array([[0.0]])
    )
    assert np.allclose(
        res_hf.eta(sign=("plus", "minus"), spin=("alpha", "alpha")), np.array([[0.0]])
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
        res_hf.eta(sign=("plus", "minus"), spin=("beta", "beta")), np.array([[0.0]])
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "plus"), spin=("alpha", "alpha")), np.array([[0.0]])
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
        res_hf.eta(sign=("minus", "plus"), spin=("beta", "beta")), np.array([[0.0]])
    )
    assert np.allclose(
        res_hf.eta(sign=("minus", "minus"), spin=("alpha", "alpha")), np.array([[0.0]])
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
        res_hf.eta(sign=("minus", "minus"), spin=("beta", "beta")), np.array([[0.0]])
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("alpha", "alpha")),
        np.array([[0.17904753]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("alpha", "beta")),
        np.array([[0.34067892]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("beta", "alpha")),
        np.array([[0.34067892]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "plus"), spin=("beta", "beta")),
        np.array([[0.17904753]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("alpha", "alpha")),
        np.array([[0.35349844]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("alpha", "beta")),
        np.array([[0.37014908]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("beta", "alpha")),
        np.array([[0.37014908]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("plus", "minus"), spin=("beta", "beta")),
        np.array([[0.35349844]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("alpha", "alpha")),
        np.array([[0.3460948]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("alpha", "beta")),
        np.array([[0.35291415]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("beta", "alpha")),
        np.array([[0.35291415]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "plus"), spin=("beta", "beta")),
        np.array([[0.3460948]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("alpha", "alpha")),
        np.array([[0.60468484]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("alpha", "beta")),
        np.array([[0.67954711]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("beta", "alpha")),
        np.array([[0.67954711]]),
    )
    assert np.allclose(
        res_lda.eta(sign=("minus", "minus"), spin=("beta", "beta")),
        np.array([[0.60468484]]),
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

    M_HF = np.zeros([140, 140], float)
    M_LDA = np.zeros([140, 140], float)
    K_l_pa_lda = np.zeros([1, 140], float)
    K_l_pb_lda = np.zeros([1, 140], float)
    K_l_ma_lda = np.zeros([1, 140], float)
    K_l_mb_lda = np.zeros([1, 140], float)
    K_l_pa_hf = np.zeros([1, 140], float)
    K_l_pb_hf = np.zeros([1, 140], float)
    K_l_ma_hf = np.zeros([1, 140], float)
    K_l_mb_hf = np.zeros([1, 140], float)
    for s in range(2):
        for i_c, i in enumerate(indices[0][s]):
            for a_c, a in enumerate(indices[1][s]):
                ias = (
                    s * len(indices[0][0]) * len(indices[1][0])
                    + i_c * len(indices[1][s])
                    + a_c
                )
                K_l_pa_lda[0][ias] = (
                    K_l_pa_lda[0][ias]
                    + two_electron_int[5, 5, i, a]
                    + molgrid.integrate(
                        f_xc_values[s]
                        * MO_values_conjugated[5]
                        * MO_values[5]
                        * MO_values_conjugated[i]
                        * MO_values[a]
                    )
                )
                K_l_ma_lda[0][ias] = (
                    K_l_ma_lda[0][ias]
                    + two_electron_int[4, 4, i, a]
                    + molgrid.integrate(
                        f_xc_values[s]
                        * MO_values_conjugated[4]
                        * MO_values[4]
                        * MO_values_conjugated[i]
                        * MO_values[a]
                    )
                )
                K_l_pb_lda[0][ias] = (
                    K_l_pb_lda[0][ias]
                    + two_electron_int[24, 24, i, a]
                    + molgrid.integrate(
                        f_xc_values[s + 1]
                        * MO_values_conjugated[24]
                        * MO_values[24]
                        * MO_values_conjugated[i]
                        * MO_values[a]
                    )
                )
                K_l_mb_lda[0][ias] = (
                    K_l_mb_lda[0][ias]
                    + two_electron_int[23, 23, i, a]
                    + molgrid.integrate(
                        f_xc_values[s + 1]
                        * MO_values_conjugated[23]
                        * MO_values[23]
                        * MO_values_conjugated[i]
                        * MO_values[a]
                    )
                )
                K_l_pa_hf[0][ias] = K_l_pa_hf[0][ias] + two_electron_int[5, 5, i, a]
                K_l_ma_hf[0][ias] = K_l_ma_hf[0][ias] + two_electron_int[4, 4, i, a]
                K_l_pb_hf[0][ias] = K_l_pb_hf[0][ias] + two_electron_int[24, 24, i, a]
                K_l_mb_hf[0][ias] = K_l_mb_hf[0][ias] + two_electron_int[23, 23, i, a]
                if s == 0:
                    K_l_pa_hf[0][ias] = K_l_pa_hf[0][ias] - two_electron_int[5, 5, i, a]
                    K_l_ma_hf[0][ias] = K_l_ma_hf[0][ias] - two_electron_int[4, 4, i, a]
                if s == 1:
                    K_l_pb_hf[0][ias] = (
                        K_l_pb_hf[0][ias] - two_electron_int[24, 24, i, a]
                    )
                    K_l_mb_hf[0][ias] = (
                        K_l_mb_hf[0][ias] - two_electron_int[23, 23, i, a]
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
    M_HF = np.linalg.inv(M_HF)
    M_LDA = np.linalg.inv(M_LDA)
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pa_lda[0, :70], M_LDA[:70, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pa_lda[0, :70], M_LDA[:70, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pa_lda[0, 70:], M_LDA[:70, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pa_lda[0, 70:], M_LDA[:70, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_a_lda = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_ma_lda[0, :70], M_LDA[:70, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_ma_lda[0, :70], M_LDA[:70, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_ma_lda[0, 70:], M_LDA[:70, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_ma_lda[0, 70:], M_LDA[:70, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_a_lda = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pb_lda[0, :70], M_LDA[70:, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pb_lda[0, :70], M_LDA[70:, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pb_lda[0, 70:], M_LDA[70:, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pb_lda[0, 70:], M_LDA[70:, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_b_lda = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_mb_lda[0, :70], M_LDA[70:, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_mb_lda[0, :70], M_LDA[70:, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_mb_lda[0, 70:], M_LDA[70:, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_mb_lda[0, 70:], M_LDA[70:, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_b_lda = np.array([a, b])

    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pa_hf[0, :70], M_HF[:70, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pa_hf[0, :70], M_HF[:70, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pa_hf[0, 70:], M_HF[:70, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pa_hf[0, 70:], M_HF[:70, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_a_hf = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_ma_hf[0, :70], M_HF[:70, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_ma_hf[0, :70], M_HF[:70, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_ma_hf[0, 70:], M_HF[:70, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_ma_hf[0, 70:], M_HF[:70, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_m_a_hf = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pb_hf[0, :70], M_HF[70:, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pb_hf[0, :70], M_HF[70:, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_pb_hf[0, 70:], M_HF[70:, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_pb_hf[0, 70:], M_HF[70:, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    fukui_p_b_hf = np.array([a, b])
    a = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_mb_hf[0, :70], M_HF[70:, :70])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_mb_hf[0, :70], M_HF[70:, :70]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
                np.zeros([len(indices[1][0]), len(indices[1][0])]),
            ],
        ]
    )
    b = -np.block(
        [
            [
                np.zeros([len(indices[0][0]), len(indices[0][0])]),
                (np.dot(K_l_mb_hf[0, 70:], M_HF[70:, 70:])).reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ],
            [
                (np.dot(K_l_mb_hf[0, 70:], M_HF[70:, 70:]))
                .reshape([len(indices[0][0]), len(indices[1][0])])
                .T,
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


def test_Response_tools_evaluate_linear_response():
    r1 = np.array(
        [
            [-2.0, -2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, 2.0, 2.0],
            [2.0, 2.0, -2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    r2 = np.array([[-0.5, -0.5, -0.5]])
    LR_hf_aa = np.array(
        [
            [2.71680112e-05],
            [3.15944035e-04],
            [1.78527933e-05],
            [2.99477727e-04],
            [2.03420969e-05],
            [4.26253754e-04],
            [1.50864226e-05],
            [6.14304675e-04],
        ]
    )
    LR_hf_ab = np.array(
        [
            [-1.62891789e-05],
            [-3.34718490e-04],
            [-1.46072300e-05],
            [-3.27078794e-04],
            [-1.69738028e-05],
            [-3.79976504e-04],
            [-1.57175914e-05],
            [-3.98667372e-04],
        ]
    )
    LR_hf_ba = np.array(
        [
            [-1.62891789e-05],
            [-3.34718490e-04],
            [-1.46072300e-05],
            [-3.27078794e-04],
            [-1.69738028e-05],
            [-3.79976504e-04],
            [-1.57175914e-05],
            [-3.98667372e-04],
        ]
    )
    LR_hf_bb = np.array(
        [
            [2.71680112e-05],
            [3.15944035e-04],
            [1.78527933e-05],
            [2.99477727e-04],
            [2.03420969e-05],
            [4.26253754e-04],
            [1.50864226e-05],
            [6.14304675e-04],
        ]
    )
    LR_lda_aa = np.array(
        [
            [1.32283211e-05],
            [-6.78128444e-06],
            [4.09714299e-06],
            [-2.45479742e-05],
            [4.33980067e-06],
            [7.45441863e-05],
            [7.90756341e-07],
            [2.72591833e-04],
        ]
    )
    LR_lda_ab = np.array(
        [
            [-2.20374695e-06],
            [-1.71282985e-05],
            [-8.30789942e-07],
            [-1.06338300e-05],
            [-1.92539329e-06],
            [-4.41526661e-05],
            [-1.21406804e-06],
            [-6.65666904e-05],
        ]
    )
    LR_lda_ba = np.array(
        [
            [-2.20374695e-06],
            [-1.71282985e-05],
            [-8.30789942e-07],
            [-1.06338300e-05],
            [-1.92539329e-06],
            [-4.41526661e-05],
            [-1.21406804e-06],
            [-6.65666904e-05],
        ]
    )
    LR_lda_bb = np.array(
        [
            [1.32283211e-05],
            [-6.78128444e-06],
            [4.09714299e-06],
            [-2.45479742e-05],
            [4.33980067e-06],
            [7.45441863e-05],
            [7.90756341e-07],
            [2.72591833e-04],
        ]
    )
    assert np.allclose(
        res_hf.evaluate_linear_response(r1, r2, spin=("alpha", "alpha")), LR_hf_aa
    )
    assert np.allclose(
        res_hf.evaluate_linear_response(r1, r2, spin=("alpha", "beta")), LR_hf_ab
    )
    assert np.allclose(
        res_hf.evaluate_linear_response(r1, r2, spin=("beta", "alpha")), LR_hf_ba
    )
    assert np.allclose(
        res_hf.evaluate_linear_response(r1, r2, spin=("beta", "beta")), LR_hf_bb
    )
    assert np.allclose(
        res_lda.evaluate_linear_response(r1, r2, spin=("alpha", "alpha")), LR_lda_aa
    )
    assert np.allclose(
        res_lda.evaluate_linear_response(r1, r2, spin=("alpha", "beta")), LR_lda_ab
    )
    assert np.allclose(
        res_lda.evaluate_linear_response(r1, r2, spin=("beta", "alpha")), LR_lda_ba
    )
    assert np.allclose(
        res_lda.evaluate_linear_response(r1, r2, spin=("beta", "beta")), LR_lda_bb
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


def test_Response_tools_evaluate_linear_response_raise():
    r1 = np.array(
        [
            [-2.0, -2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, 2.0, 2.0],
            [2.0, 2.0, -2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    r2 = np.array([[-0.5, -0.5, -0.5]])
    r3 = "hello"
    r4 = np.array([1])
    r5 = np.array([[1, 2, 3, 4]])
    with pytest.raises(TypeError) as error:
        et = res_lda.evaluate_linear_response(r1, r2, spin=1)
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r1, r2, spin=("a", "b", "c"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r1, r2, spin=("alpha", "b"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r1, r2, spin=("a", "beta"))
    assert (
        str(error.value)
        == """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
    )
    with pytest.raises(TypeError) as error:
        et = res_lda.evaluate_linear_response(r3, r2, spin=("alpha", "alpha"))
    assert str(error.value) == """'r1' must be a np.ndarray"""
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r4, r2, spin=("alpha", "alpha"))
    assert str(error.value) == """'r1' must be a np.ndarray with shape (N, 3)"""
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r5, r2, spin=("alpha", "alpha"))
    assert str(error.value) == """'r1' must be a np.ndarray with shape (N, 3)"""
    with pytest.raises(TypeError) as error:
        et = res_lda.evaluate_linear_response(r1, r3, spin=("alpha", "alpha"))
    assert str(error.value) == """'r2' must be a np.ndarray"""
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r1, r4, spin=("alpha", "alpha"))
    assert str(error.value) == """'r2' must be a np.ndarray with shape (N, 3)"""
    with pytest.raises(ValueError) as error:
        et = res_lda.evaluate_linear_response(r1, r5, spin=("alpha", "alpha"))
    assert str(error.value) == """'r2' must be a np.ndarray with shape (N, 3)"""
