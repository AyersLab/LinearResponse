import numpy as np
import pytest
from M_matrix import M_matrix
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
M_mat = M_matrix(basis, molecule, coord_type="cartesian")
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


def test_M_matrix_M_block_size():
    assert M_mat.M_block_size == 70


def test_M_matrix_M_s():
    M_s = np.zeros([140, 140], float)
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
                            if s == t and i == j and a == b:
                                M_s[ias][jbt] = (
                                    M_s[ias][jbt]
                                    + molecule.mo.energies[a]
                                    - molecule.mo.energies[i]
                                )
    assert np.allclose(M_mat.M_s(), M_s)


def test_M_matrix_calculate_M():
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
    assert np.allclose(M_mat.calculate_M(k="HF", complex=False), M_HF)
    assert np.allclose(
        M_mat.calculate_M(k="lda_x", complex=False, molgrid=molgrid), M_LDA
    )


def test_M_matrix_LR_Excitations():
    E_hf = np.array(
        [
            0.2762758,
            0.36803327,
            0.39799405,
            0.45971039,
            0.47930996,
            0.49034119,
            0.50788335,
            0.53074219,
            0.5737651,
            0.58105451,
            0.63788014,
            0.75112764,
            0.88639542,
            1.08608281,
            1.09137113,
            1.09210694,
            1.10129602,
            1.13583064,
            1.15182422,
            1.1577442,
            1.19094255,
            1.19847869,
            1.20576961,
            1.21818314,
            1.22988867,
            1.23834541,
            1.24229888,
            1.28002114,
            1.28541526,
            1.28857599,
            1.30592244,
            1.32802361,
            1.34649719,
            1.37771964,
            1.39500888,
            1.4090743,
            1.42329799,
            1.43579374,
            1.4451964,
            1.45163306,
            1.46246249,
            1.46686219,
            1.4726919,
            1.49819733,
            1.50701933,
            1.54641346,
            1.55551389,
            1.58398846,
            1.61801674,
            1.70973668,
            1.73920061,
            1.76774085,
            1.77440979,
            1.84781045,
            1.85809326,
            1.89913972,
            1.91321444,
            1.94979251,
            1.95526076,
            1.95827312,
            1.97582005,
            1.99501781,
            2.00452955,
            2.0638323,
            2.07229266,
            2.07249069,
            2.08473492,
            2.12402264,
            2.13609952,
            2.13950369,
            2.15820319,
            2.16098224,
            2.19083631,
            2.22665481,
            2.2312452,
            2.23551656,
            2.27365687,
            2.30438847,
            2.30571898,
            2.31921589,
            2.32890879,
            2.37546034,
            2.53528383,
            2.65699607,
            2.69224333,
            2.69988555,
            2.70847025,
            2.72340366,
            2.75668401,
            2.77827141,
            2.82082144,
            2.83417917,
            2.86265259,
            2.8685378,
            2.88415694,
            2.91303843,
            2.92748869,
            3.05245068,
            3.07190404,
            3.17762121,
            3.28142264,
            3.3605723,
            3.57279005,
            3.66143019,
            3.85144985,
            3.91747883,
            3.92560699,
            4.00954903,
            4.08495392,
            4.13174471,
            4.55026784,
            4.66301609,
            19.27572465,
            19.30752826,
            19.36524843,
            19.38880715,
            19.99932582,
            20.00313549,
            20.06484843,
            20.08179202,
            20.10867252,
            20.12168693,
            20.16168798,
            20.17647309,
            20.25745248,
            20.35370201,
            20.41113569,
            20.44211518,
            20.94645554,
            20.94735444,
            20.96729876,
            20.96818034,
            20.99955083,
            21.00044212,
            21.50757324,
            21.50898208,
            21.80408706,
            21.80520173,
            22.6452125,
            22.94014885,
        ]
    )
    E_lda = np.array(
        [
            0.37469979,
            0.40073991,
            0.44180983,
            0.44974421,
            0.47103459,
            0.48627623,
            0.53726699,
            0.57776251,
            0.59207524,
            0.6460131,
            0.66342366,
            0.76689943,
            1.07334812,
            1.08747936,
            1.10236167,
            1.14075776,
            1.14776958,
            1.15510444,
            1.17707461,
            1.17888957,
            1.19681643,
            1.21195168,
            1.21351133,
            1.2234259,
            1.24587752,
            1.2508285,
            1.26947616,
            1.2726155,
            1.28682815,
            1.29986487,
            1.30770382,
            1.32404047,
            1.34414938,
            1.39158653,
            1.40684015,
            1.42440176,
            1.42816674,
            1.44013113,
            1.44076856,
            1.44642736,
            1.46416207,
            1.48846325,
            1.48956814,
            1.49425607,
            1.55453522,
            1.56561878,
            1.56966599,
            1.57661327,
            1.60590092,
            1.72184044,
            1.76951492,
            1.80083819,
            1.80498414,
            1.88037535,
            1.90441993,
            1.91338453,
            1.92541575,
            1.95762126,
            1.97027289,
            1.97988634,
            2.00959869,
            2.02786788,
            2.04309229,
            2.06624971,
            2.07803283,
            2.09097686,
            2.11893002,
            2.12410627,
            2.1244647,
            2.14776107,
            2.16090173,
            2.16113842,
            2.20656066,
            2.21517903,
            2.23162518,
            2.2382194,
            2.29177515,
            2.31845706,
            2.34498262,
            2.34757261,
            2.36689135,
            2.39231082,
            2.55792393,
            2.64057292,
            2.68951442,
            2.6980826,
            2.69983667,
            2.75394759,
            2.78525358,
            2.79218183,
            2.80593731,
            2.81720742,
            2.86686277,
            2.87245482,
            2.89428687,
            2.89893287,
            2.9378852,
            3.0792901,
            3.08774021,
            3.19654008,
            3.28792275,
            3.36821235,
            3.580232,
            3.66891469,
            3.83857389,
            3.904826,
            3.91412188,
            3.99993193,
            4.07484725,
            4.12157367,
            4.54597296,
            4.66643994,
            19.25186764,
            19.28892777,
            19.37386186,
            19.39568692,
            20.0006302,
            20.00384468,
            20.04749787,
            20.09484227,
            20.11523437,
            20.11613861,
            20.14060145,
            20.20983386,
            20.29091828,
            20.38932976,
            20.41665635,
            20.44722336,
            20.94569048,
            20.94658122,
            20.96667229,
            20.96755351,
            20.99964404,
            21.00053277,
            21.50637103,
            21.50776838,
            21.80418777,
            21.80530024,
            22.23299014,
            22.52518481,
        ]
    )
    assert np.allclose(M_mat.Excitations_energies_real_MO(k="HF"), E_hf)
    assert np.allclose(
        M_mat.Excitations_energies_real_MO(k="lda_x", molgrid=molgrid), E_lda
    )


def test_M_matrix_raise_calculate_M():
    with pytest.raises(TypeError) as error:
        M_ma = M_mat.calculate_M(k=1)
    assert str(error.value) == """'k' must be None or a str"""
    with pytest.raises(TypeError) as error:
        M_ma = M_mat.calculate_M(k="lda_x", molgrid=1)
    assert str(error.value) == """'molgrid' must be None or a 'MolGrid' instance"""
    with pytest.raises(TypeError) as error:
        M_ma = M_mat.calculate_M(k="lda_x", molgrid=molgrid, complex=1)
    assert str(error.value) == """'complex' must be a bool"""
    with pytest.raises(TypeError) as error:
        M_ma = M_mat.calculate_M(k="lda_x", molgrid=molgrid, inverse=1)
    assert str(error.value) == """'inverse' must be a bool"""
    with pytest.raises(ValueError) as error:
        M_ma = M_mat.calculate_M(k="abcd", molgrid=molgrid)
    assert (
        str(error.value)
        == """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
    )


def test_M_matrix_raise_Excitations_energies_real_MO():
    with pytest.raises(TypeError) as error:
        Ene = M_mat.Excitations_energies_real_MO(k=1)
    assert str(error.value) == """'k' must be None or a str"""
    with pytest.raises(TypeError) as error:
        Ene = M_mat.Excitations_energies_real_MO(k="lda_x", molgrid=1)
    assert str(error.value) == """'molgrid' must be None or a 'MolGrid' instance"""
    with pytest.raises(ValueError) as error:
        Ene = M_mat.Excitations_energies_real_MO(k="abcd", molgrid=molgrid)
    assert (
        str(error.value)
        == """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
    )
