import numpy as np
import warnings
from M_matrix import M_matrix


class Response_tools(M_matrix):
    """Class for the analytical tools matrices"""

    def __init__(
        self,
        basis,
        molecule,
        coord_type="spherical",
        k=None,
        molgrid=None,
        conjugate=False,
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
        conjugate : Boolean, default is False
            If true, compute (K)ias,bjt and (K)ias,jbt
            (K)ias,jbt and (K)ias,bjt can differ only in case of complex MO

        Raises
        ------
        TypeError
            If 'k' is not None or a str
            If 'molgrid' is not a 'MolGrid' instance
            If 'conjugate' is not a bool
        ValueError
            If 'k' is not 'HF' or a supported functional code"""

        super().__init__(basis, molecule, coord_type)
        self._k = k
        self._molgrid = molgrid
        self._conjugate = conjugate
        self._M_inv = None
        self._K_l_p_a = None
        self._K_l_m_a = None
        self._K_l_p_b = None
        self._K_l_m_b = None
        self._K_p_p_a = None
        self._K_p_m_a = None
        self._K_p_p_b = None
        self._K_p_m_b = None

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
            self._M_inv = self.calculate_M(
                k=self._k,
                molgrid=self._molgrid,
                conjugate=self._conjugate,
                inverse=True,
            )
        return self._M_inv

    @property
    def K_line_p_a(self):
        if not isinstance(self._K_l_p_a, np.ndarray):
            index = self.Frontier_MO_index(sign="plus", spin="alpha")
            if self._k == "HF":
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_HF(
                    shape="line", index=index
                )
                if self._conjugate == True:
                    self._K_l_p_a = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_HF(shape="line", index=index, conjugate=True),
                        ]
                    )
                else:
                    self._K_l_p_a = np.array([K_ffsjbt, K_ffsjbt])
            elif isinstance(self._k, str):
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    shape="line",
                    index=index,
                )
                if self._conjugate == True:
                    self._K_l_p_a = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_DFT(
                                XC_functional=self._k,
                                molgrid=self._molgrid,
                                shape="line",
                                index=index,
                                conjugate=True,
                            ),
                        ]
                    )
                else:
                    self._K_l_p_a = np.array([K_ffsjbt, K_ffsjbt])
            else:
                K_ffsjbt = np.zeros([1, self.M_size])
                self._K_l_p_a = np.array([K_ffsjbt, K_ffsjbt])
        return self._K_l_p_a

    @property
    def K_line_m_a(self):
        if not isinstance(self._K_l_m_a, np.ndarray):
            index = self.Frontier_MO_index(sign="minus", spin="alpha")
            if self._k == "HF":
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_HF(
                    shape="line", index=index
                )
                if self._conjugate == True:
                    self._K_l_m_a = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_HF(shape="line", index=index, conjugate=True),
                        ]
                    )
                else:
                    self._K_l_m_a = np.array([K_ffsjbt, K_ffsjbt])
            elif isinstance(self._k, str):
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    shape="line",
                    index=index,
                )
                if self._conjugate == True:
                    self._K_l_m_a = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_DFT(
                                XC_functional=self._k,
                                molgrid=self._molgrid,
                                shape="line",
                                index=index,
                                conjugate=True,
                            ),
                        ]
                    )
                else:
                    self._K_l_m_a = np.array([K_ffsjbt, K_ffsjbt])
            else:
                K_ffsjbt = np.zeros([1, self.M_size])
                self._K_l_m_a = np.array([K_ffsjbt, K_ffsjbt])
        return self._K_l_m_a

    @property
    def K_line_p_b(self):
        if not isinstance(self._K_l_p_b, np.ndarray):
            index = self.Frontier_MO_index(sign="plus", spin="beta")
            if self._k == "HF":
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_HF(
                    shape="line", index=index
                )
                if self._conjugate == True:
                    self._K_l_p_b = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_HF(shape="line", index=index, conjugate=True),
                        ]
                    )
                else:
                    self._K_l_p_b = np.array([K_ffsjbt, K_ffsjbt])
            elif isinstance(self._k, str):
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    shape="line",
                    index=index,
                )
                if self._conjugate == True:
                    self._K_l_p_b = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_DFT(
                                XC_functional=self._k,
                                molgrid=self._molgrid,
                                shape="line",
                                index=index,
                                conjugate=True,
                            ),
                        ]
                    )
                else:
                    self._K_l_p_b = np.array([K_ffsjbt, K_ffsjbt])
            else:
                K_ffsjbt = np.zeros([1, self.M_size])
                self._K_l_p_b = np.array([K_ffsjbt, K_ffsjbt])
        return self._K_l_p_b

    @property
    def K_line_m_b(self):
        if not isinstance(self._K_l_m_b, np.ndarray):
            index = self.Frontier_MO_index(sign="minus", spin="beta")
            if self._k == "HF":
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_HF(
                    shape="line", index=index
                )
                if self._conjugate == True:
                    self._K_l_m_b = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_HF(shape="line", index=index, conjugate=True),
                        ]
                    )
                else:
                    self._K_l_m_b = np.array([K_ffsjbt, K_ffsjbt])
            elif isinstance(self._k, str):
                K_ffsjbt = self.K_coulomb(shape="line", index=index) + self.K_fxc_DFT(
                    XC_functional=self._k,
                    molgrid=self._molgrid,
                    shape="line",
                    index=index,
                )
                if self._conjugate == True:
                    self._K_l_m_b = np.array(
                        [
                            K_ffsjbt,
                            self.K_coulomb(shape="line", index=index, conjugate=True)
                            + self.K_fxc_DFT(
                                XC_functional=self._k,
                                molgrid=self._molgrid,
                                shape="line",
                                index=index,
                                conjugate=True,
                            ),
                        ]
                    )
                else:
                    self._K_l_m_b = np.array([K_ffsjbt, K_ffsjbt])
            else:
                K_ffsjbt = np.zeros([1, self.M_size])
                self._K_l_m_b = np.array([K_ffsjbt, K_ffsjbt])
        return self._K_l_m_b

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
        index_1 = self.Frontier_MO_index(sign=sign[0], spin=spin[0])
        index_2 = self.Frontier_MO_index(sign=sign[1], spin=spin[1])
        index = [index_1[0] + index_2[0], index_1[1] + index_2[1]]
        if self._k == "HF":
            k = self.K_coulomb(shape="point", index=index) + self.K_fxc_HF(
                shape="point", index=index
            )
        elif isinstance(self._k, str):
            k = self.K_coulomb(shape="point", index=index) + self.K_fxc_DFT(
                XC_functional=self._k, molgrid=self._molgrid, shape="point", index=index
            )
        else:
            k = 0
        if sign[0] == "plus":
            if spin[0] == "alpha":
                if sign[1] == "plus":
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_p_a[0] + self.K_line_p_a[1], self.M_inverse
                            ).dot(self.K_line_p_a[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_p_b[0] + self.K_line_p_b[1], self.M_inverse
                            ).dot(self.K_line_p_a[0].T)
                            + k
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_m_a[0] + self.K_line_m_a[1], self.M_inverse
                            ).dot(self.K_line_p_a[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_m_b[0] + self.K_line_m_b[1], self.M_inverse
                            ).dot(self.K_line_p_a[0].T)
                            + k
                        )
            else:
                if sign[1] == "plus":
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_p_a[0] + self.K_line_p_a[1], self.M_inverse
                            ).dot(self.K_line_p_b[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_p_b[0] + self.K_line_p_b[1], self.M_inverse
                            ).dot(self.K_line_p_b[0].T)
                            + k
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_m_a[0] + self.K_line_m_a[1], self.M_inverse
                            ).dot(self.K_line_p_b[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_m_b[0] + self.K_line_m_b[1], self.M_inverse
                            ).dot(self.K_line_p_b[0].T)
                            + k
                        )
        else:
            if spin[0] == "alpha":
                if sign[1] == "plus":
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_p_a[0] + self.K_line_p_a[1], self.M_inverse
                            ).dot(self.K_line_m_a[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_p_b[0] + self.K_line_p_b[1], self.M_inverse
                            ).dot(self.K_line_m_a[0].T)
                            + k
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_m_a[0] + self.K_line_m_a[1], self.M_inverse
                            ).dot(self.K_line_m_a[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_m_b[0] + self.K_line_m_b[1], self.M_inverse
                            ).dot(self.K_line_m_a[0].T)
                            + k
                        )
            else:
                if sign[1] == "plus":
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_p_a[0] + self.K_line_p_a[1], self.M_inverse
                            ).dot(self.K_line_m_b[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_p_b[0] + self.K_line_p_b[1], self.M_inverse
                            ).dot(self.K_line_m_b[0].T)
                            + k
                        )
                else:
                    if spin[1] == "alpha":
                        eta = (
                            -np.dot(
                                self.K_line_m_a[0] + self.K_line_m_a[1], self.M_inverse
                            ).dot(self.K_line_m_b[0].T)
                            + k
                        )
                    else:
                        eta = (
                            -np.dot(
                                self.K_line_m_b[0] + self.K_line_m_b[1], self.M_inverse
                            ).dot(self.K_line_m_b[0].T)
                            + k
                        )
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
        if spin[0] == "alpha":
            if sign == "plus":
                K_ffsjbt = self.K_line_p_a[0]
                K_ffsbjt = self.K_line_p_a[1]
            else:
                K_ffsjbt = self.K_line_m_a[0]
                K_ffsbjt = self.K_line_m_a[1]
        else:
            if sign == "plus":
                K_ffsjbt = self.K_line_p_b[0]
                K_ffsbjt = self.K_line_p_b[1]
            else:
                K_ffsjbt = self.K_line_m_b[0]
                K_ffsbjt = self.K_line_m_b[1]
        indices = self.K_indices()
        a = np.block(
            [
                [
                    np.zeros([len(indices[0][0]), len(indices[0][0])]),
                    (
                        np.dot(K_ffsjbt + K_ffsbjt, self.M_inverse)[
                            :, : int(self.M_size / 2)
                        ]
                    ).reshape([len(indices[0][0]), len(indices[1][0])]),
                ],
                [
                    (
                        np.dot(K_ffsjbt + K_ffsbjt, self.M_inverse)[
                            :, : int(self.M_size / 2)
                        ]
                    ).reshape([len(indices[1][0]), len(indices[0][0])]),
                    np.zeros([len(indices[1][0]), len(indices[1][0])]),
                ],
            ]
        )
        b = np.block(
            [
                [
                    np.zeros([len(indices[0][0]), len(indices[0][0])]),
                    (
                        np.dot(K_ffsjbt + K_ffsbjt, self.M_inverse)[
                            :, int(self.M_size / 2) :
                        ]
                    ).reshape([len(indices[0][0]), len(indices[1][0])]),
                ],
                [
                    (
                        np.dot(K_ffsjbt + K_ffsbjt, self.M_inverse)[
                            :, int(self.M_size / 2) :
                        ]
                    ).reshape([len(indices[1][0]), len(indices[0][0])]),
                    np.zeros([len(indices[1][0]), len(indices[1][0])]),
                ],
            ]
        )
        fukui = np.array([a, b])
        if spin[0] == "alpha":
            fukui[
                0,
                self.Frontier_MO_index(sign=sign, spin="alpha")[0],
                self.Frontier_MO_index(sign=sign, spin="alpha")[0],
            ] = 1
        else:
            fukui[
                1,
                self.Frontier_MO_index(sign=sign, spin="beta")[1][0]
                - self._molecule.mo.nbasis,
                self.Frontier_MO_index(sign=sign, spin="beta")[1][0]
                - self._molecule.mo.nbasis,
            ] = 1
        if spin[1] == "alpha":
            return fukui[0]
        else:
            return fukui[1]

    def linear_response(self, spin):
        """Return linear response matrices

        Parameters
        ----------
        spin: tuple of two str,
            elements can only be 'alpha' or 'beta'

        Return
        ------
        linear_response: np.ndarray"""
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
        Mat = -2 * self.M_inverse
        linear_response = np.array(
            [
                Mat[: int(self.M_size / 2)][:, : int(self.M_size / 2)],
                Mat[: int(self.M_size / 2)][:, int(self.M_size / 2) :],
                Mat[int(self.M_size / 2) :][:, : int(self.M_size / 2)],
                Mat[int(self.M_size / 2) :][:, int(self.M_size / 2) :],
            ]
        )
        if spin == ("alpha", "alpha"):
            return linear_response[0]
        elif spin == ("alpha", "beta"):
            return linear_response[1]
        elif spin == ("beta", "alpha"):
            return linear_response[2]
        else:
            return linear_response[3]
