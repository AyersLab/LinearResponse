import numpy as np
import pytest
from gbasis.integrals.point_charge import point_charge_integral
from Alchemical_tools import Alchemical_tools
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

molecule = load_one("h2o.fchk")
basis = from_iodata(molecule)
q = np.array([0.1])
R = np.array([[2.574156, -0.181816, -2.453822]])
alc_hf = Alchemical_tools(
    basis,
    molecule,
    point_charge_positions=R,
    point_charges_values=q,
    coord_type="cartesian",
    k="HF",
)
r0 = 1e-10
radii = get_cov_radii(molecule.atnums)
n_rad_points = 100
deg = 5
onedgrid = GaussChebyshev(n_rad_points)
molecule_at_grid = []
for i in range(len(molecule.atnums)):
    Rad = BeckeTF.find_parameter(onedgrid.points, r0, radii[i])
    rad_grid = BeckeTF(r0, Rad).transform_grid(onedgrid)
    molecule_at_grid = molecule_at_grid + [
        AtomicGrid.special_init(rad_grid, radii[i], degs=[deg], scales=[])
    ]
molgrid = MolGrid(molecule_at_grid, molecule.atnums)
alc_lda = Alchemical_tools(
    basis,
    molecule,
    point_charge_positions=R,
    point_charges_values=q,
    coord_type="cartesian",
    k="lda_x",
    molgrid=molgrid,
)
two_electron_int = electron_repulsion_integral(
    basis, transform=molecule.mo.coeffs.T, coord_type="cartesian", notation="chemist"
)
xc_function = pylibxc.LibXCFunctional("lda_x", "polarized")
a_dm = np.dot(molecule.mo.coeffsa * molecule.mo.occsa, molecule.mo.coeffsa.T.conj())
b_dm = np.dot(molecule.mo.coeffsb * molecule.mo.occsb, molecule.mo.coeffsb.T.conj())
a_density_values = evaluate_density(a_dm, basis, molgrid.points, coord_type="cartesian")
b_density_values = evaluate_density(b_dm, basis, molgrid.points, coord_type="cartesian")
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
M_HF_inv = np.linalg.inv(M_HF)
M_LDA_inv = np.linalg.inv(M_LDA)


def test_Alchemical_tools_density_matrix_variation():

    PT_CHARGE_INT = point_charge_integral(
        basis, R, q, transform=molecule.mo.coeffs.T, coord_type="cartesian"
    )
    dP_hf = np.array([np.diag(molecule.mo.occsa), np.diag(molecule.mo.occsb)])
    dP_hf = 0 * dP_hf
    dP_lda = 0 * dP_hf
    for s in range(2):
        for i_c, i in enumerate(occupied_ind[s]):
            for a_c, a in enumerate(virtual_ind[s]):
                ias = (
                    s * len(occupied_ind[0]) * len(virtual_ind[0])
                    + i_c * len(virtual_ind[s])
                    + a_c
                )
                for t in range(2):
                    for j_c, j in enumerate(occupied_ind[t]):
                        for b_c, b in enumerate(virtual_ind[t]):
                            jbt = (
                                t * len(occupied_ind[0]) * len(virtual_ind[0])
                                + j_c * len(virtual_ind[t])
                                + b_c
                            )
                            dv_ias = PT_CHARGE_INT[a][i][0]
                            if t == 0:
                                dP_hf[t][j][b] = (
                                    dP_hf[t][j][b] - M_HF_inv[ias][jbt] * dv_ias
                                )
                                dP_hf[t][b][j] = (
                                    dP_hf[t][b][j] - M_HF_inv[ias][jbt] * dv_ias
                                )
                                dP_lda[t][j][b] = (
                                    dP_lda[t][j][b] - M_LDA_inv[ias][jbt] * dv_ias
                                )
                                dP_lda[t][b][j] = (
                                    dP_lda[t][b][j] - M_LDA_inv[ias][jbt] * dv_ias
                                )
                            if t == 1:
                                dP_hf[t][j - 19][b - 19] = (
                                    dP_hf[t][j - 19][b - 19]
                                    - M_HF_inv[ias][jbt] * dv_ias
                                )
                                dP_hf[t][b - 19][j - 19] = (
                                    dP_hf[t][b - 19][j - 19]
                                    - M_HF_inv[ias][jbt] * dv_ias
                                )
                                dP_lda[t][j - 19][b - 19] = (
                                    dP_lda[t][j - 19][b - 19]
                                    - M_LDA_inv[ias][jbt] * dv_ias
                                )
                                dP_lda[t][b - 19][j - 19] = (
                                    dP_lda[t][b - 19][j - 19]
                                    - M_LDA_inv[ias][jbt] * dv_ias
                                )
    assert np.allclose(alc_hf.density_matrix_variation(), dP_hf)
    assert np.allclose(alc_lda.density_matrix_variation(), dP_lda)


def test_Alchemical_tools_energy_variation():
    PT_CHARGE_INT = point_charge_integral(
        basis, R, q, transform=molecule.mo.coeffs.T, coord_type="cartesian"
    )
    dE_1 = 0.0
    for i_c, i in enumerate(molecule.mo.occs):
        if i != 0:
            dE_1 = dE_1 + PT_CHARGE_INT[i_c][i_c][0]
    dE_2_hf = 0.0
    dE_2_lda = 0.0
    for s in range(2):
        for i_c, i in enumerate(occupied_ind[s]):
            for a_c, a in enumerate(virtual_ind[s]):
                ias = (
                    s * len(occupied_ind[0]) * len(virtual_ind[0])
                    + i_c * len(virtual_ind[s])
                    + a_c
                )
                for t in range(2):
                    for j_c, j in enumerate(occupied_ind[t]):
                        for b_c, b in enumerate(virtual_ind[t]):
                            jbt = (
                                t * len(occupied_ind[0]) * len(virtual_ind[0])
                                + j_c * len(virtual_ind[t])
                                + b_c
                            )
                            dv_ias = PT_CHARGE_INT[a][i][0]
                            dv_jbt = PT_CHARGE_INT[b][j][0]
                            dE_2_hf = dE_2_hf - dv_ias * dv_jbt * M_HF_inv[ias][jbt]
                            dE_2_lda = dE_2_lda - dv_ias * dv_jbt * M_LDA_inv[ias][jbt]
    assert np.allclose(alc_hf.energy_variation(), np.array([dE_1, dE_2_hf]))
    assert np.allclose(alc_lda.energy_variation(), np.array([dE_1, dE_2_lda]))


def test_Alchemical_tools_init_raise():
    with pytest.raises(TypeError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=R,
            point_charges_values=1,
            coord_type="cartesian",
            k="HF",
        )
    assert str(error.value) == """'point_charges_values' must be a np.ndarray"""
    with pytest.raises(TypeError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=1,
            point_charges_values=q,
            coord_type="cartesian",
            k="HF",
        )
    assert str(error.value) == """'point_charge_positions' must be a np.ndarray"""
    with pytest.raises(ValueError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=R,
            point_charges_values=np.array([[1, 2], [3, 4]]),
            coord_type="cartesian",
            k="HF",
        )
    assert (
        str(error.value)
        == """'point_charges_values' must be a np.ndarray with shape (N,)"""
    )
    with pytest.raises(ValueError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=np.array([1, 2]),
            point_charges_values=q,
            coord_type="cartesian",
            k="HF",
        )
    assert (
        str(error.value)
        == """'point_charge_positions' must be a np.ndarray with shape (N, 3)"""
    )
    with pytest.raises(ValueError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=np.array([[1], [2], [3]]),
            point_charges_values=q,
            coord_type="cartesian",
            k="HF",
        )
    assert (
        str(error.value)
        == """'point_charge_positions' must be a np.ndarray with shape (N, 3)"""
    )
    with pytest.raises(ValueError) as error:
        alc = Alchemical_tools(
            basis,
            molecule,
            point_charge_positions=np.array([[1, 2, 3]]),
            point_charges_values=np.array([1, 2]),
            coord_type="cartesian",
            k="HF",
        )
    assert (
        str(error.value)
        == """'point_charge_positions' and 'point_charges_values' must have matching shapes"""
    )
