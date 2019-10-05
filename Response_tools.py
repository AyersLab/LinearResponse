import numpy as np
import warnings
from M_matrix import M_matrix
from gbasis.evals.eval import evaluate_basis


class Response_tools(M_matrix):
    """Class for the analytical tools matrices"""

    def __init__(
        self,
        basis,
        molecule,
        coord_type="spherical",
        k=None,
        molgrid=None,
        complex=False,
    ):
        """Initialise the Response tools class

        Parameters
        ----------
        k: str or None
            The coupling matrix K type
            If None : No coupling matrix, Independent particle approximation
            If 'HF', K is the Hartree-Fock coupling matrix
            If 'Functional_code', K is DFT the coupling matrix associated with the XC functional
            corresponding to 'Functional_code' which can be found at:
            https://tddft.org/programs/libxc/functionals/
            Only the LDA and GGA in the list are supported
        molgrid: MolGrid class object (from grid package)  suitable for numerical integration
            of any real space function related to the molecule (such as the density)
            Necessary for any DFT coupling matrices
        Complex : Boolean, default is False
            If true, consider that MO have complex values

        Raises
        ------
        TypeError
            If 'k' is not None or a str
            If 'molgrid' is not a 'MolGrid' instance
            If 'complex' is not a bool
        ValueError
            If 'k' is not 'HF' or a supported functional code"""

        super().__init__(basis, molecule, coord_type)
        self._k = k
        self._molgrid = molgrid
        self._complex = complex
        self._basis = basis
        self._molecule = molecule
        self._coord_type = coord_type
        self._M_inv = None

    def Frontier_MO_index(self, sign, spin):
        """Return the frontier MO indices

        Parameters
        ----------
        sign: str,
            can only be 'plus' or 'minus'
        spin: str,
            can only be 'alpha' or 'beta'

        Return
        ------
        index: list of two list of index
            [[], [beta frontier orbital]] or [[alpha frontier orbital], []]

        Raises
        ------
        ValueError
            If sign is not 'plus' or 'minus'"""
        if not isinstance(spin, str):
            raise TypeError("""'spin' must be 'alpha' or 'beta'""")
        if not isinstance(sign, str):
            raise TypeError("""'sign' must be 'plus' or 'minus'""")
        if not spin in ["alpha", "beta"]:
            raise ValueError("""'spin' must be 'alpha' or 'beta'""")
        if not sign in ["plus", "minus"]:
            raise ValueError("""'sign' must be 'plus' or 'minus'""")
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in true_divide"
        )
        if sign == "plus":
            index = [
                np.nanargmax(
                    1 / (self._molecule.mo.energiesa * (1 - self._molecule.mo.occsa))
                ),
                self._molecule.mo.nbasis
                + np.nanargmax(
                    1 / (self._molecule.mo.energiesb * (1 - self._molecule.mo.occsb))
                ),
            ]
        else:
            index = [
                np.nanargmin(
                    1 / (self._molecule.mo.energiesa * (self._molecule.mo.occsa))
                ),
                self._molecule.mo.nbasis
                + np.nanargmin(
                    1 / (self._molecule.mo.energiesb * (self._molecule.mo.occsb))
                ),
            ]
        if spin == "alpha":
            return [[index[0]], []]
        else:
            return [[], [index[1]]]

    @property
    def M_inverse(self):
        if not isinstance(self._M_inv, np.ndarray):
            M = self.calculate_M(
                k=self._k, molgrid=self._molgrid, complex=self._complex, inverse=True
            )
            b = self.M_block_size
            if self._complex == True:
                self._M_inv = np.array(
                    [
                        [
                            M[0 * b : 1 * b, 0 * b : 1 * b],
                            M[0 * b : 1 * b, 1 * b : 2 * b],
                            M[0 * b : 1 * b, 2 * b : 3 * b],
                            M[0 * b : 1 * b, 3 * b : 4 * b],
                        ],
                        [
                            M[1 * b : 2 * b, 0 * b : 1 * b],
                            M[1 * b : 2 * b, 1 * b : 2 * b],
                            M[1 * b : 2 * b, 2 * b : 3 * b],
                            M[1 * b : 2 * b, 3 * b : 4 * b],
                        ],
                        [
                            M[2 * b : 3 * b, 0 * b : 1 * b],
                            M[2 * b : 3 * b, 1 * b : 2 * b],
                            M[2 * b : 3 * b, 2 * b : 3 * b],
                            M[2 * b : 3 * b, 3 * b : 4 * b],
                        ],
                        [
                            M[3 * b : 4 * b, 0 * b : 1 * b],
                            M[3 * b : 4 * b, 1 * b : 2 * b],
                            M[3 * b : 4 * b, 2 * b : 3 * b],
                            M[3 * b : 4 * b, 3 * b : 4 * b],
                        ],
                    ]
                )
            else:
                self._M_inv = np.array(
                    [
                        [
                            M[0 * b : 1 * b, 0 * b : 1 * b],
                            M[0 * b : 1 * b, 1 * b : 2 * b],
                        ],
                        [
                            M[1 * b : 2 * b, 0 * b : 1 * b],
                            M[1 * b : 2 * b, 1 * b : 2 * b],
                        ],
                    ]
                )
        return self._M_inv

    def mhu(self, sign, spin):
        """Return the chemical potential

        Parameters
        ----------
        sign: str
            can only be 'plus' or 'minus'
        spin: str
            can only be 'alpha' or 'beta'

        Return
        ------
        mhu: float
            mhu_alpha or mhu_beta"""
        index = self.Frontier_MO_index(sign=sign, spin=spin)
        if spin == "alpha":
            return self._molecule.mo.energies[index[0]]
        else:
            return self._molecule.mo.energies[index[1]]

    def eta(self, sign, spin):
        """Calculate the Hardness

        Parameters
        ----------
        sign: tuple of two str
            list elements can only be 'plus' or 'minus'
        spin: tuple of two str
            list elements can only be 'alpha' or 'beta'

        Return
        ------
        eta: 1D ndarray
            eta^{sign}_{spin}

        Raises
        ------
        ValueError
            If 'spin' is not a tuple of two elements being either 'alpha' or 'beta'
            If 'sign' is not a tuple of two elements being either 'plus' or 'minus'"""
        if not isinstance(spin, tuple):
            raise TypeError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if len(spin) != 2:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[0] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[1] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not isinstance(sign, tuple):
            raise TypeError(
                """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
            )
        if len(sign) != 2:
            raise ValueError(
                """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
            )
        if not sign[0] in ["plus", "minus"]:
            raise ValueError(
                """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
            )
        if not sign[1] in ["plus", "minus"]:
            raise ValueError(
                """'sign' must be a tuple of two str, being either 'plus' or 'minus'"""
            )
        index_l1_a = self.Frontier_MO_index(sign=sign[0], spin="alpha")
        index_l1_b = self.Frontier_MO_index(sign=sign[0], spin="beta")
        index_l2_a = self.Frontier_MO_index(sign=sign[1], spin="alpha")
        index_l2_b = self.Frontier_MO_index(sign=sign[1], spin="beta")
        index_pt = [
            self.Frontier_MO_index(sign=sign[0], spin=spin[0])[0]
            + self.Frontier_MO_index(sign=sign[1], spin=spin[1])[0],
            self.Frontier_MO_index(sign=sign[0], spin=spin[0])[1]
            + self.Frontier_MO_index(sign=sign[1], spin=spin[1])[1],
        ]
        M_inv = self.M_inverse
        if self._complex == True:
            if self._k == "HF":
                K_pt = self.K_fxc_HF(
                    Type=1, shape="point", index=index_pt
                ) + self.K_coulomb(Type=1, shape="point", index=index_pt)
                K_l2_1_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l2_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_a)
                K_l2_1_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l2_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_b)
                K_l2_2_a = self.K_fxc_HF(
                    Type=2, shape="line", index=index_l2_a
                ) + self.K_coulomb(Type=2, shape="line", index=index_l2_a)
                K_l2_2_b = self.K_fxc_HF(
                    Type=2, shape="line", index=index_l2_b
                ) + self.K_coulomb(Type=2, shape="line", index=index_l2_b)
                K_l1_1_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l1_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_a)
                K_l1_1_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l1_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_b)
                K_l1_2_a = self.K_fxc_HF(
                    Type=2, shape="line", index=index_l1_a
                ) + self.K_coulomb(Type=2, shape="line", index=index_l1_a)
                K_l1_2_b = self.K_fxc_HF(
                    Type=2, shape="line", index=index_l1_b
                ) + self.K_coulomb(Type=2, shape="line", index=index_l1_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 0])
                                + K_l2_2_a[0].dot(M_inv[1, 0])
                                + K_l2_1_b[2].dot(M_inv[2, 0])
                                + K_l2_2_b[2].dot(M_inv[3, 0])
                            ).dot(K_l1_2_a[0].T)
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 1])
                                + K_l2_2_a[0].dot(M_inv[1, 1])
                                + K_l2_1_b[2].dot(M_inv[2, 1])
                                + K_l2_2_b[2].dot(M_inv[3, 1])
                            ).dot(K_l1_1_a[0].T)
                        )
                    else:
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 2])
                                + K_l2_2_a[1].dot(M_inv[1, 2])
                                + K_l2_1_b[3].dot(M_inv[2, 2])
                                + K_l2_2_b[3].dot(M_inv[3, 2])
                            ).dot(K_l1_2_a[1].T)
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 3])
                                + K_l2_2_a[1].dot(M_inv[1, 3])
                                + K_l2_1_b[3].dot(M_inv[2, 3])
                                + K_l2_2_b[3].dot(M_inv[3, 3])
                            ).dot(K_l1_1_a[1].T)
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 0])
                                + K_l2_2_a[0].dot(M_inv[1, 0])
                                + K_l2_1_b[2].dot(M_inv[2, 0])
                                + K_l2_2_b[2].dot(M_inv[3, 0])
                            ).dot(K_l1_2_b[2].T)
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 1])
                                + K_l2_2_a[0].dot(M_inv[1, 1])
                                + K_l2_1_b[2].dot(M_inv[2, 1])
                                + K_l2_2_b[2].dot(M_inv[3, 1])
                            ).dot(K_l1_1_b[2].T)
                        )
                    else:
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 2])
                                + K_l2_2_a[1].dot(M_inv[1, 2])
                                + K_l2_1_b[3].dot(M_inv[2, 2])
                                + K_l2_2_b[3].dot(M_inv[3, 2])
                            ).dot(K_l1_2_b[3].T)
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 3])
                                + K_l2_2_a[1].dot(M_inv[1, 3])
                                + K_l2_1_b[3].dot(M_inv[2, 3])
                                + K_l2_2_b[3].dot(M_inv[3, 3])
                            ).dot(K_l1_1_b[3].T)
                        )
            elif isinstance(self._k, str):
                K_pt = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="point",
                    index=index_pt,
                ) + self.K_coulomb(Type=1, shape="point", index=index_pt)
                K_l2_1_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l2_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_a)
                K_l2_1_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l2_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_b)
                K_l2_2_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l2_a,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l2_a)
                K_l2_2_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l2_b,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l2_b)
                K_l1_1_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l1_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_a)
                K_l1_1_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l1_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_b)
                K_l1_2_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l1_a,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l1_a)
                K_l1_2_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l1_b,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l1_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 0])
                                + K_l2_2_a[0].dot(M_inv[1, 0])
                                + K_l2_1_b[2].dot(M_inv[2, 0])
                                + K_l2_2_b[2].dot(M_inv[3, 0])
                            ).dot(K_l1_2_a[0].T)
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 1])
                                + K_l2_2_a[0].dot(M_inv[1, 1])
                                + K_l2_1_b[2].dot(M_inv[2, 1])
                                + K_l2_2_b[2].dot(M_inv[3, 1])
                            ).dot(K_l1_1_a[0].T)
                        )
                    else:
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 2])
                                + K_l2_2_a[1].dot(M_inv[1, 2])
                                + K_l2_1_b[3].dot(M_inv[2, 2])
                                + K_l2_2_b[3].dot(M_inv[3, 2])
                            ).dot(K_l1_2_a[1].T)
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 3])
                                + K_l2_2_a[1].dot(M_inv[1, 3])
                                + K_l2_1_b[3].dot(M_inv[2, 3])
                                + K_l2_2_b[3].dot(M_inv[3, 3])
                            ).dot(K_l1_1_a[1].T)
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 0])
                                + K_l2_2_a[0].dot(M_inv[1, 0])
                                + K_l2_1_b[2].dot(M_inv[2, 0])
                                + K_l2_2_b[2].dot(M_inv[3, 0])
                            ).dot(K_l1_2_b[2].T)
                            - (
                                K_l2_1_a[0].dot(M_inv[0, 1])
                                + K_l2_2_a[0].dot(M_inv[1, 1])
                                + K_l2_1_b[2].dot(M_inv[2, 1])
                                + K_l2_2_b[2].dot(M_inv[3, 1])
                            ).dot(K_l1_1_b[2].T)
                        )
                    else:
                        eta = (
                            K_pt
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 2])
                                + K_l2_2_a[1].dot(M_inv[1, 2])
                                + K_l2_1_b[3].dot(M_inv[2, 2])
                                + K_l2_2_b[3].dot(M_inv[3, 2])
                            ).dot(K_l1_2_b[3].T)
                            - (
                                K_l2_1_a[1].dot(M_inv[0, 3])
                                + K_l2_2_a[1].dot(M_inv[1, 3])
                                + K_l2_1_b[3].dot(M_inv[2, 3])
                                + K_l2_2_b[3].dot(M_inv[3, 3])
                            ).dot(K_l1_1_b[3].T)
                        )
            else:
                eta = None
        else:
            if self._k == "HF":
                K_pt = self.K_fxc_HF(
                    Type=1, shape="point", index=index_pt
                ) + self.K_coulomb(Type=1, shape="point", index=index_pt)
                K_l2_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l2_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_a)
                K_l2_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l2_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_b)
                K_l1_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l1_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_a)
                K_l1_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l1_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        eta = K_pt - 2 * (
                            K_l2_a[0].dot(M_inv[0, 0]) + K_l2_b[2].dot(M_inv[1, 0])
                        ).dot(K_l1_a[0].T)
                    else:
                        eta = K_pt - 2 * (
                            K_l2_a[1].dot(M_inv[0, 1]) + K_l2_b[3].dot(M_inv[1, 1])
                        ).dot(K_l1_a[1].T)
                else:
                    if spin[1] == "alpha":
                        eta = K_pt - 2 * (
                            K_l2_a[0].dot(M_inv[0, 0]) + K_l2_b[2].dot(M_inv[1, 0])
                        ).dot(K_l1_b[2].T)
                    else:
                        eta = K_pt - 2 * (
                            K_l2_a[1].dot(M_inv[0, 1]) + K_l2_b[3].dot(M_inv[1, 1])
                        ).dot(K_l1_b[3].T)
            elif isinstance(self._k, str):
                K_pt = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="point",
                    index=index_pt,
                ) + self.K_coulomb(Type=1, shape="point", index=index_pt)
                K_l2_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l2_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_a)
                K_l2_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l2_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l2_b)
                K_l1_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l1_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_a)
                K_l1_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l1_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l1_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        eta = K_pt - 2 * (
                            K_l2_a[0].dot(M_inv[0, 0]) + K_l2_b[2].dot(M_inv[1, 0])
                        ).dot(K_l1_a[0].T)
                    else:
                        eta = K_pt - 2 * (
                            K_l2_a[1].dot(M_inv[0, 1]) + K_l2_b[3].dot(M_inv[1, 1])
                        ).dot(K_l1_a[1].T)
                else:
                    if spin[1] == "alpha":
                        eta = K_pt - 2 * (
                            K_l2_a[0].dot(M_inv[0, 0]) + K_l2_b[2].dot(M_inv[1, 0])
                        ).dot(K_l1_b[2].T)
                    else:
                        eta = K_pt - 2 * (
                            K_l2_a[1].dot(M_inv[0, 1]) + K_l2_b[3].dot(M_inv[1, 1])
                        ).dot(K_l1_b[3].T)
            else:
                eta = None
        return eta

    def fukui(self, sign, spin):
        """Calculate the fukui matrices

        Parameters
        ----------
        sign: str,
            can only be 'plus' or 'minus'
        spin: tuple of two str,
            tuple elements can only be 'alpha' or 'beta'

        Return
        ------
        fukui: 3D ndarray shape = (4, molecule.mo.nbasis, molecule.mo.nbasis)
            fukui[0] is the spin_alpha fukui matrix
            fukui[1] is the spin_beta fukui matrix

        Raises
        ------
        ValueError
            If 'spin' is not a tuple of 'alpha' or 'beta'
            If 'sign' is not 'plus' or 'minus'"""
        if not isinstance(spin, tuple):
            raise TypeError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if len(spin) != 2:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[0] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[1] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not isinstance(sign, str):
            raise TypeError("""'sign' must be 'plus' or 'minus'""")
        if not sign in ["plus", "minus"]:
            raise ValueError("""'sign' must be 'plus' or 'minus'""")
        index_l_a = self.Frontier_MO_index(sign=sign, spin="alpha")
        index_l_b = self.Frontier_MO_index(sign=sign, spin="beta")
        indices = self.K_indices()
        M_inv = self.M_inverse
        if self._complex == True:
            if self._k == "HF":
                K_l_1_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_a)
                K_l_1_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_b)
                K_l_2_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_a)
                K_l_2_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        ab_block = -(
                            K_l_1_a[0].dot(M_inv[0, 0]) + K_l_2_a[0].dot(M_inv[1, 0])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_a[0].dot(M_inv[0, 1]) + K_l_2_a[0].dot(M_inv[1, 1]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                    else:
                        ab_block = -(
                            K_l_1_a[1].dot(M_inv[0, 2]) + K_l_2_a[1].dot(M_inv[1, 2])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_a[1].dot(M_inv[0, 3]) + K_l_2_a[1].dot(M_inv[1, 3]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                else:
                    if spin[1] == "alpha":
                        ab_block = -(
                            K_l_1_b[2].dot(M_inv[2, 0]) + K_l_2_b[2].dot(M_inv[3, 0])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_b[2].dot(M_inv[2, 1]) + K_l_2_b[2].dot(M_inv[3, 1]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                    else:
                        ab_block = -(
                            K_l_1_b[3].dot(M_inv[2, 2]) + K_l_2_b[3].dot(M_inv[3, 2])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_b[3].dot(M_inv[2, 3]) + K_l_2_b[3].dot(M_inv[3, 3]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
            elif isinstance(self._k, str):
                K_l_1_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_a)
                K_l_1_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_b)
                K_l_2_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l_a,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l_a)
                K_l_2_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=2,
                    shape="line",
                    index=index_l_b,
                ) + self.K_coulomb(Type=2, shape="line", index=index_l_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        ab_block = -(
                            K_l_1_a[0].dot(M_inv[0, 0]) + K_l_2_a[0].dot(M_inv[1, 0])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_a[0].dot(M_inv[0, 1]) + K_l_2_a[0].dot(M_inv[1, 1]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                    else:
                        ab_block = -(
                            K_l_1_a[1].dot(M_inv[0, 2]) + K_l_2_a[1].dot(M_inv[1, 2])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_a[1].dot(M_inv[0, 3]) + K_l_2_a[1].dot(M_inv[1, 3]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                else:
                    if spin[1] == "alpha":
                        ab_block = -(
                            K_l_1_b[2].dot(M_inv[2, 0]) + K_l_2_b[2].dot(M_inv[3, 0])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_b[2].dot(M_inv[2, 1]) + K_l_2_b[2].dot(M_inv[3, 1]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
                    else:
                        ab_block = -(
                            K_l_1_b[3].dot(M_inv[2, 2]) + K_l_2_b[3].dot(M_inv[3, 2])
                        ).reshape([len(indices[0][0]), len(indices[1][0])])
                        ba_block = (
                            -(K_l_1_b[3].dot(M_inv[2, 3]) + K_l_2_b[3].dot(M_inv[3, 3]))
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                            .T
                        )
            else:
                ab_block = np.zeros([len(indices[0][0]), len(indices[1][0])])
                ba_block = np.zeros([len(indices[1][0]), len(indices[0][0])])
        else:
            if self._k == "HF":
                K_l_a = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_a
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_a)
                K_l_b = self.K_fxc_HF(
                    Type=1, shape="line", index=index_l_b
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        ab_block = (
                            -K_l_a[0]
                            .dot(M_inv[0, 0])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                    else:
                        ab_block = (
                            -K_l_a[1]
                            .dot(M_inv[0, 1])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                else:
                    if spin[1] == "alpha":
                        ab_block = (
                            -K_l_b[2]
                            .dot(M_inv[1, 0])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                    else:
                        ab_block = (
                            -K_l_b[3]
                            .dot(M_inv[1, 1])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                ba_block = ab_block.T
            elif isinstance(self._k, str):
                K_l_a = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l_a,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_a)
                K_l_b = self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    Type=1,
                    shape="line",
                    index=index_l_b,
                ) + self.K_coulomb(Type=1, shape="line", index=index_l_b)
                if spin[0] == "alpha":
                    if spin[1] == "alpha":
                        ab_block = (
                            -K_l_a[0]
                            .dot(M_inv[0, 0])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                    else:
                        ab_block = (
                            -K_l_a[1]
                            .dot(M_inv[0, 1])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                else:
                    if spin[1] == "alpha":
                        ab_block = (
                            -K_l_b[2]
                            .dot(M_inv[1, 0])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                    else:
                        ab_block = (
                            -K_l_b[3]
                            .dot(M_inv[1, 1])
                            .reshape([len(indices[0][0]), len(indices[1][0])])
                        )
                ba_block = ab_block.T
            else:
                ab_block = np.zeros([len(indices[0][0]), len(indices[1][0])])
                ba_block = ab_block.T
        fukui = np.block(
            [
                [np.zeros([len(indices[0][0]), len(indices[0][0])]), ab_block],
                [ba_block, np.zeros([len(indices[1][0]), len(indices[1][0])])],
            ]
        )
        if spin[0] == spin[1] == "alpha":
            fukui[index_l_a[0][0], index_l_a[0][0]] = 1
        if spin[0] == spin[1] == "beta":
            fukui[
                index_l_b[1][0] - self._molecule.mo.nbasis,
                index_l_b[1][0] - self._molecule.mo.nbasis,
            ] = 1
        return fukui

    def evaluate_linear_response(self, r1, r2, spin):
        """Return linear response matrices

        Parameters
        ----------
        spin: tuple of two str,
            elements can only be 'alpha' or 'beta'
        r1: ndarray
            N1 coordinates
        r2: ndarray
            N2 coordiantes

        Return
        ------
        linear_response: np.ndarray of shape [N1, N2]"""
        if not isinstance(spin, tuple):
            raise TypeError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if len(spin) != 2:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[0] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not spin[1] in ["alpha", "beta"]:
            raise ValueError(
                """'spin' must be a tuple of two str, being either 'alpha' or 'beta'"""
            )
        if not isinstance(r1, np.ndarray):
            raise TypeError("""'r1' must be a np.ndarray""")
        if len(r1.shape) != 2:
            raise ValueError("""'r1' must be a np.ndarray with shape (N, 3)""")
        if r1.shape[1] != 3:
            raise ValueError("""'r1' must be a np.ndarray with shape (N, 3)""")
        if not isinstance(r2, np.ndarray):
            raise TypeError("""'r2' must be a np.ndarray""")
        if len(r2.shape) != 2:
            raise ValueError("""'r2' must be a np.ndarray with shape (N, 3)""")
        if r2.shape[1] != 3:
            raise ValueError("""'r2' must be a np.ndarray with shape (N, 3)""")
        indices = self.K_indices()
        MO_r1 = evaluate_basis(
            self._basis,
            r1,
            transform=self._molecule.mo.coeffs.T,
            coord_type=self._coord_type,
        )
        MO_r2 = evaluate_basis(
            self._basis,
            r2,
            transform=self._molecule.mo.coeffs.T,
            coord_type=self._coord_type,
        )
        a_occ_r1 = MO_r1[indices[0][0]]
        a_occ_r2 = MO_r2[indices[0][0]]
        b_occ_r1 = MO_r1[indices[0][1]]
        b_occ_r2 = MO_r2[indices[0][1]]
        a_virt_r1 = MO_r1[indices[1][0]]
        a_virt_r2 = MO_r2[indices[1][0]]
        b_virt_r1 = MO_r1[indices[1][1]]
        b_virt_r2 = MO_r2[indices[1][1]]
        del (MO_r1, MO_r2)
        phi_a_r1 = a_occ_r1[:, None, :] * a_virt_r1[None, :, :]
        phi_a_r1 = phi_a_r1.reshape(
            phi_a_r1.shape[0] * phi_a_r1.shape[1], phi_a_r1.shape[2]
        )
        phi_a_r2 = a_occ_r2[:, None, :] * a_virt_r2[None, :, :]
        phi_a_r2 = phi_a_r2.reshape(
            phi_a_r2.shape[0] * phi_a_r2.shape[1], phi_a_r2.shape[2]
        )
        del (a_occ_r1, a_occ_r2, a_virt_r1, a_virt_r2)
        phi_b_r1 = b_occ_r1[:, None, :] * b_virt_r1[None, :, :]
        phi_b_r1 = phi_b_r1.reshape(
            phi_b_r1.shape[0] * phi_b_r1.shape[1], phi_b_r1.shape[2]
        )
        phi_b_r2 = b_occ_r2[:, None, :] * b_virt_r2[None, :, :]
        phi_b_r2 = phi_b_r2.reshape(
            phi_b_r2.shape[0] * phi_b_r2.shape[1], phi_b_r2.shape[2]
        )
        del (b_occ_r1, b_occ_r2, b_virt_r1, b_virt_r2)
        M_inv = self.M_inverse
        if self._complex == True:
            if spin[0] == "alpha":
                if spin[1] == "alpha":
                    linear_response = (
                        np.dot(phi_a_r1.conj().T, M_inv[0, 0]).dot(phi_a_r2)
                        + np.dot(phi_a_r1.conj().T, M_inv[0, 1]).dot(phi_a_r2.conj())
                        + np.dot(phi_a_r1.T, M_inv[1, 0]).dot(phi_a_r2)
                        + np.dot(phi_a_r1.T, M_inv[1, 1]).dot(phi_a_r2.conj())
                    )
                else:
                    linear_response = (
                        np.dot(phi_a_r1.conj().T, M_inv[0, 2]).dot(phi_b_r2)
                        + np.dot(phi_a_r1.conj().T, M_inv[0, 3]).dot(phi_b_r2.conj())
                        + np.dot(phi_a_r1.T, M_inv[1, 2]).dot(phi_b_r2)
                        + np.dot(phi_a_r1.T, M_inv[1, 3]).dot(phi_b_r2.conj())
                    )
            else:
                if spin[1] == "alpha":
                    linear_response = (
                        np.dot(phi_b_r1.conj().T, M_inv[2, 0]).dot(phi_a_r2)
                        + np.dot(phi_b_r1.conj().T, M_inv[2, 1]).dot(phi_a_r2.conj())
                        + np.dot(phi_b_r1.T, M_inv[3, 0]).dot(phi_a_r2)
                        + np.dot(phi_b_r1.T, M_inv[3, 1]).dot(phi_a_r2.conj())
                    )
                else:
                    linear_response = (
                        np.dot(phi_b_r1.conj().T, M_inv[2, 2]).dot(phi_b_r2)
                        + np.dot(phi_b_r1.conj().T, M_inv[2, 3]).dot(phi_b_r2.conj())
                        + np.dot(phi_b_r1.T, M_inv[3, 2]).dot(phi_b_r2)
                        + np.dot(phi_b_r1.T, M_inv[3, 3]).dot(phi_b_r2.conj())
                    )
        else:
            if spin[0] == "alpha":
                if spin[1] == "alpha":
                    linear_response = -2 * (
                        np.dot(phi_a_r1.T, M_inv[0, 0]).dot(phi_a_r2)
                    )
                else:
                    linear_response = -2 * (
                        np.dot(phi_a_r1.T, M_inv[0, 1]).dot(phi_b_r2)
                    )
            else:
                if spin[1] == "alpha":
                    linear_response = -2 * (
                        np.dot(phi_b_r1.T, M_inv[1, 0]).dot(phi_a_r2)
                    )
                else:
                    linear_response = -2 * (
                        np.dot(phi_b_r1.T, M_inv[1, 1]).dot(phi_b_r2)
                    )
        return linear_response
