import numpy as np
from numpy import linalg as la
from K_matrices import K_matrices
import pylibxc
from grid.molgrid import MolGrid


class M_matrix(K_matrices):
    """Class for the M matrices"""

    @property
    def M_block_size(self):
        """Calculate the size of the M matrix

        Return
        ------
        M_block_size: int"""
        M_block_size = self.K_shape(shape="square")[0]
        return M_block_size

    def M_s(self, complex=False):
        """"Calculate the non interacting M matrix

        Parameters
        ----------
        complex: bool, default is False

        Return
        ------
        ndarray

        Raises
        ------
        TypeError
            If complex is not a boolean"""
        if not isinstance(complex, bool):
            raise TypeError("""'complex' must be a bool""")
        indices = self.K_indices(shape="square")
        M_s = np.diag(
            np.array(
                [
                    (
                        self._molecule.mo.energies[indices[1][0]][:, None]
                        - self._molecule.mo.energies[indices[0][0]]
                    ).T.reshape(len(indices[1][0]) * len(indices[0][0])),
                    (
                        self._molecule.mo.energies[indices[1][1]][:, None]
                        - self._molecule.mo.energies[indices[0][1]]
                    ).T.reshape(len(indices[1][1]) * len(indices[0][1])),
                ]
            ).reshape(2*self.M_block_size)
        )
        if complex == False:
            return M_s
        else:
            return np.block(
                [
                    [
                        M_s[: self.M_block_size, : self.M_block_size],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                    ],
                    [
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, : self.M_block_size],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                    ],
                    [
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[self.M_block_size :, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                    ],
                    [
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[: self.M_block_size, self.M_block_size :],
                        M_s[self.M_block_size :, self.M_block_size :],
                    ],
                ]
            )

    def calculate_M(self, k=None, molgrid=None, complex=False, inverse=False):
        """Calculate the M matrix

        Parameters
        ----------
        k: str or None
            The coupling matrix K type
            If None : No coupling matrix, Independent particle approximation
            If 'HF', K is the Hartree-Fock coupling matrix
            If 'Functional_code', K is DFT the coupling matrix associated with the XC functional
            corresponding to 'Functional_code' which can be found at:
            https://tddft.org/programs/libxc/functionals/
            All the LDA and GGA in the list are supported
        molgrid: MolGrid class object (from grid package)  suitable for numerical integration
            of any real space function related to the molecule (such as the density)
            Necessary for any DFT coupling matrices
        complex : bool, default is False
        inverse: bool, default is False
            If True, return the inverse of the matrix M

        Return
        ------
        M: ndarray(M_size, M_size)
            The M matrix (or its inverse depending on inverse option)

        Raises
        ------
        TypeError
            If 'k' is not None or a str
            If 'molgrid' is not a 'MolGrid' instance
            If 'type' is not a bool
            If 'inverse' is not a bool
        ValueError
            If 'k' is not 'HF' or a supported functional code"""
        if k != None and not isinstance(k, str):
            raise TypeError("""'k' must be None or a str""")
        if molgrid != None and not isinstance(molgrid, MolGrid):
            raise TypeError("""'molgrid' must be None or a 'MolGrid' instance""")
        if not isinstance(complex, bool):
            raise TypeError("""'complex' must be a bool""")
        if not isinstance(inverse, bool):
            raise TypeError("""'inverse' must be a bool""")
        if not k in pylibxc.util.xc_available_functional_names() + ["HF"]:
            raise ValueError(
                """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
            )
        M = self.M_s(complex = complex)
        b = self.M_block_size
        if complex == True:
            if k == "HF":
                K_1 = self.K_coulomb(Type=1, shape = 'square') + self.K_fxc_HF(Type=1, shape='square')
                K_2 = self.K_coulomb(Type=2, shape = 'square') + self.K_fxc_HF(Type=2, shape='square')
            elif isinstance(k, str):
                K_1 = self.K_coulomb(Type=1, shape = 'square') + self.K_fxc_DFT(molgrid = molgrid, Type =1, XC_functional=k, shape = 'square')
                K_2 = self.K_coulomb(Type=2, shape = 'square') + self.K_fxc_DFT(molgrid = molgrid, Type =2, XC_functional=k, shape = 'square')
            M[0 * b:1 * b, 0 * b:1 * b] = M[0 * b:1 * b, 0 * b:1 * b] + K_1[0]
            M[0 * b:1 * b, 1 * b:2 * b] = M[0 * b:1 * b, 1 * b:2 * b] + K_2[0]
            M[0 * b:1 * b, 2 * b:3 * b] = M[0 * b:1 * b, 2 * b:3 * b] + K_1[1]
            M[0 * b:1 * b, 3 * b:4 * b] = M[0 * b:1 * b, 3 * b:4 * b] + K_2[1]

            M[1 * b:2 * b, 0 * b:1 * b] = M[1 * b:2 * b, 0 * b:1 * b] + K_2[0].conj()
            M[1 * b:2 * b, 1 * b:2 * b] = M[1 * b:2 * b, 1 * b:2 * b] + K_1[0].conj()
            M[1 * b:2 * b, 2 * b:3 * b] = M[1 * b:2 * b, 2 * b:3 * b] + K_2[1].conj()
            M[1 * b:2 * b, 3 * b:4 * b] = M[1 * b:2 * b, 3 * b:4 * b] + K_1[1].conj()

            M[2 * b:3 * b, 0 * b:1 * b] = M[2 * b:3 * b, 0 * b:1 * b] + K_1[2]
            M[2 * b:3 * b, 1 * b:2 * b] = M[2 * b:3 * b, 1 * b:2 * b] + K_2[2]
            M[2 * b:3 * b, 2 * b:3 * b] = M[2 * b:3 * b, 2 * b:3 * b] + K_1[3]
            M[2 * b:3 * b, 3 * b:4 * b] = M[2 * b:3 * b, 3 * b:4 * b] + K_2[3]

            M[3 * b:4 * b, 0 * b:1 * b] = M[3 * b:4 * b, 0 * b:1 * b] + K_2[2].conj()
            M[3 * b:4 * b, 1 * b:2 * b] = M[3 * b:4 * b, 1 * b:2 * b] + K_1[2].conj()
            M[3 * b:4 * b, 2 * b:3 * b] = M[3 * b:4 * b, 2 * b:3 * b] + K_2[3].conj()
            M[3 * b:4 * b, 3 * b:4 * b] = M[3 * b:4 * b, 3 * b:4 * b] + K_1[3].conj()
        else:
            if k == "HF":
                K = self.K_coulomb(Type=1, shape = 'square') + self.K_fxc_HF(Type=1, shape='square')
            elif isinstance(k, str):
                K = self.K_coulomb(Type=1, shape = 'square') + self.K_fxc_DFT(molgrid = molgrid, Type =1, XC_functional=k, shape = 'square')
            M[0 * b:1 * b, 0 * b:1 * b] = M[0 * b:1 * b, 0 * b:1 * b] + 2 * K[0]
            M[0 * b:1 * b, 1 * b:2 * b] = M[0 * b:1 * b, 1 * b:2 * b] + 2 * K[1]

            M[1 * b:2 * b, 0 * b:1 * b] = M[0 * b:1 * b, 2 * b:3 * b] + 2 * K[2]
            M[1 * b:2 * b, 1 * b:2 * b] = M[0 * b:1 * b, 3 * b:4 * b] + 2 * K[3]
        if inverse == True:
            M = la.inv(M)
        return M

    def Excitations_energies_real_MO(self, k=None, molgrid=None):
        """Calculate the excitation energies

        Parameters
        ----------
        k: str or None
            The coupling matrix K type
            If None : No coupling matrix, Independent particle approximation
            If 'HF', K is the Hartree-Fock coupling matrix
            If 'Functional_code', K is DFT the coupling matrix associated with the XC functional
            corresponding to 'Functional_code' which can be found at:
            https://tddft.org/programs/libxc/functionals/
            All the LDA and GGA in the list are supported
        molgrid: MolGrid class object (from grid package)  suitable for numerical integration
            of any real space function related to the molecule (such as the density)
            Necessary for any DFT coupling matrices

        Return
        ------
        Exci: dict
            'Excitations': ndarray shape = M_size
            'Transition densities: ndarray shape = [M_size, M_size]

        Raises
        ------
        TypeError
            If 'k' is not None or a str
            If 'molgrid' is not a 'MolGrid' instance
        ValueError
            If 'k' is not 'HF' or a supported functional code"""
        if k != None and not isinstance(k, str):
            raise TypeError("""'k' must be None or a str""")
        if molgrid != None and not isinstance(molgrid, MolGrid):
            raise TypeError("""'molgrid' must be None or a 'MolGrid' instance""")
        if not k in pylibxc.util.xc_available_functional_names() + ["HF"]:
            raise ValueError(
                """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
            )
        M = self.calculate_M(k = k, molgrid = molgrid)
        Omega = np.dot(np.sqrt(self.M_s(complex=False)),M).dot(np.sqrt(self.M_s(complex=False)))
        Excitations = la.eigvalsh(Omega)
        return np.sqrt(Excitations)
()