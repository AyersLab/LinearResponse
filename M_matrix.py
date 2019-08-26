import numpy as np
from numpy import linalg as la
from K_matrices import K_matrices
import pylibxc
from grid.molgrid import MolGrid


class M_matrix(K_matrices):
    """Class for the M matrices"""

    @property
    def M_size(self):
        """Calculate the size of the M matrix

        Return
        ------
        M_size: int"""
        M_size = self.K_shape(shape="square")[0]
        return M_size

    def M_s(self):
        """"Calculate the non interacting M matrix

        Return
        ------
        ndarray : Diagonal matrix of shape (M_size, M_size)"""
        indices = self.K_indices(shape="square")
        M_s_numpy = np.diag(
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
            ).reshape(self.M_size)
        )

        return M_s_numpy

    def calculate_M(self, k=None, molgrid=None, conjugate=False, inverse=False):
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
        conjugate : Boolean, default is False
            If true compute (K_coulomb)ias,bjt and (K_coulomb)ias,jbt
            (K_coulomb)ias,jbt and (K_coulomb)ias,bjt can differ only in case of complex MO
        inverse: Boolean, default is False
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
            If 'conjugate' is not a bool
            If 'inverse' is not a bool
        ValueError
            If 'k' is not 'HF' or a supported functional code"""
        if k != None and not isinstance(k, str):
            raise TypeError("""'k' must be None or a str""")
        if molgrid != None and not isinstance(molgrid, MolGrid):
            raise TypeError("""'molgrid' must be None or a 'MolGrid' instance""")
        if not isinstance(conjugate, bool):
            raise TypeError("""'conjugate' must be a bool""")
        if not isinstance(inverse, bool):
            raise TypeError("""'inverse' must be a bool""")
        if not k in pylibxc.util.xc_available_functional_names() + ["HF"]:
            raise ValueError(
                """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
            )
        M = self.M_s()
        if k == "HF":
            K = self.K_coulomb() + self.K_fxc_HF()
            if conjugate == True:
                M = (
                    M
                    + K
                    + self.K_coulomb(conjugate=True)
                    + self.K_fxc_HF(conjugate=True)
                )
            else:
                M = M + 2 * K
        elif isinstance(k, str):
            K = self.K_coulomb() + self.K_fxc_DFT(XC_functional=k, molgrid=molgrid)
            if conjugate == True:
                M = (
                    M
                    + K
                    + self.K_coulomb(conjugate=True)
                    + self.K_fxc_DFT(XC_functional=k, molgrid=molgrid, conjugate=True)
                )
            else:
                M = M + 2 * K
        if inverse == True:
            M = la.inv(M)
        return M

    def LR_Excitations(self, k=None, conjugate=False, molgrid=None):
        """Calculate the excittion energies and the corresponding transition densities

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
        conjugate : Boolean, default is False
            If true calculate (K)ias,bjt and (K)ias,jbt
            (K)ias,jbt and (K)ias,bjt can differ only in case of complex MO

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
            If 'conjugate' is not a bool
        ValueError
            If 'k' is not 'HF' or a supported functional code"""
        if k != None and not isinstance(k, str):
            raise TypeError("""'k' must be None or a str""")
        if molgrid != None and not isinstance(molgrid, MolGrid):
            raise TypeError("""'molgrid' must be None or a 'MolGrid' instance""")
        if not isinstance(conjugate, bool):
            raise TypeError("""'conjugate' must be a bool""")
        if not k in pylibxc.util.xc_available_functional_names() + ["HF"]:
            raise ValueError(
                """'k' must be 'HF' of a supported functional code, fro them, see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/"""
            )
        if k == "HF":
            K_iasjbt = self.K_coulomb() + self.K_fxc_HF()
            if conjugate == True:
                K_iasbjt = self.K_coulomb(conjugate=True) + self.K_fxc_HF(
                    conjugate=True
                )
            else:
                K_iasbjt = K_iasjbt
        elif isinstance(k, str):
            K_iasjbt = self.K_coulomb() + self.K_fxc_DFT(
                XC_functional=k, molgrid=molgrid
            )
            if conjugate == True:
                K_iasbjt = self.K_coulomb(conjugate=True) + self.K_fxc_DFT(
                    XC_functional=k, molgrid=molgrid, conjugate=True
                )
            else:
                K_iasbjt = K_iasjbt
        else:
            K_iasjbt = np.zeros([self.M_size, self.M_size])
            K_iasbjt = np.zeros([self.M_size, self.M_size])

        Omega = (self.M_s() * self.M_s()) + np.dot(
            np.sqrt(self.M_s()), K_iasjbt + K_iasbjt
        ).dot(np.sqrt(self.M_s()))
        Excitations = la.eigvalsh(Omega)
        return np.sqrt(Excitations)
