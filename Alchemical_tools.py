import numpy as np
from M_matrix import M_matrix
from gbasis.integrals.point_charge import point_charge_integral


class Alchemical_tools(M_matrix):
    """Class for the analytical tools matrices"""

    def __init__(
        self,
        basis,
        molecule,
        point_charges_values,
        point_charge_positions,
        coord_type="spherical",
        k=None,
        molgrid=None,
        complex=False,
    ):
        """Initialise the Point_charge_perturbation class

        Parameters
        ----------
        point_charges_values: np.ndarray(N)
            The charges values
        point_charge_positions: np.ndarray(N, 3)
            The charges coordinates in cartesian basis, with atomic units
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
        complex : Bool, default is False
            If true, considers that MO have complex values

        Raises
        ------
        TypeError
            If 'point_charges_values' is not a np.ndarray
            if 'point_charge_positions' is not a np.ndarray
        ValueError
            If point_charges_values is not a 1D np.ndarray
            If point_charge_positions is not a 2D np.ndarray
            If point_charge_positions.shape[1] != 3
            If point_charges_values.shape[0] != point_charge_positions.shep[0]"""
        if not isinstance(point_charges_values, np.ndarray):
            raise TypeError("""'point_charges_values' must be a np.ndarray""")
        if not isinstance(point_charge_positions, np.ndarray):
            raise TypeError("""'point_charge_positions' must be a np.ndarray""")
        if len(point_charges_values.shape) != 1:
            raise ValueError(
                """'point_charges_values' must be a np.ndarray with shape (N,)"""
            )
        if len(point_charge_positions.shape) != 2:
            raise ValueError(
                """'point_charge_positions' must be a np.ndarray with shape (N, 3)"""
            )
        if point_charge_positions.shape[1] != 3:
            raise ValueError(
                """'point_charge_positions' must be a np.ndarray with shape (N, 3)"""
            )
        if point_charges_values.shape[0] != point_charge_positions.shape[0]:
            raise ValueError(
                """'point_charge_positions' and 'point_charges_values' must have matching shapes"""
            )
        super().__init__(basis, molecule, coord_type=coord_type)
        self._k = k
        self._molgrid = molgrid
        self._complex = complex
        self._M_inv = None
        self._val = point_charges_values
        self._pos = point_charge_positions
        if complex == True:
            raise ValueError("""The case of complex MO is not supported foor the alchemical tool""")

    @property
    def M_inverse(self):
        if not isinstance(self._M_inv, np.ndarray):
            self._M_inv = self.calculate_M(
                k=self._k, molgrid=self._molgrid, complex=self._complex, inverse=True
            )
        return self._M_inv


    def density_matrix_variation(self):
        """Return density matrix variation in MO basis

        Return
        ------
        dP: 3D ndarray
            dP[0] is the alpha density matrix variation
            dP[1] is the beta density matrix variation"""
        M_inv = self.M_inverse
        dv = np.sum(
            point_charge_integral(
                self._basis,
                self._pos,
                self._val,
                transform=self._molecule.mo.coeffs.T,
                coord_type=self._coord_type,
            ),
            axis=-1,
        )
        indices = self.K_indices()
        dv = np.array(
            [
                dv[indices[0][0]][:, indices[1][0]].reshape([self.M_block_size]),
                dv[indices[0][1]][:, indices[1][1]].reshape([self.M_block_size]),
            ]
        ).reshape([2*self.M_block_size])
        dP = -np.dot(M_inv, dv)
        dP = np.array(
            [
                dP[: self.M_block_size].reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
                dP[self.M_block_size :].reshape(
                    [len(indices[0][0]), len(indices[1][0])]
                ),
            ]
        )
        dP = np.array(
            [
                np.block(
                    [
                        [np.zeros([len(indices[0][0]), len(indices[0][0])]), dP[0]],
                        [dP[0].T, np.zeros([len(indices[1][0]), len(indices[1][0])])],
                    ]
                ),
                np.block(
                    [
                        [np.zeros([len(indices[0][1]), len(indices[0][1])]), dP[1]],
                        [dP[1].T, np.zeros([len(indices[1][1]), len(indices[1][1])])],
                    ]
                ),
            ]
        )
        return dP

    def energy_variation(self):
        """Return  the variation of the MO energies

        Return
        ------
        dE: 2D ndarray
            dE[0] is the first order energy variation
            dE[1] is the second order energy variation"""
        M_inv = self.M_inverse
        dv = np.sum(
            point_charge_integral(
                self._basis,
                self._pos,
                self._val,
                transform=self._molecule.mo.coeffs.T,
                coord_type=self._coord_type,
            ),
            axis=-1,
        )
        indices = self.K_indices()
        dE = np.sum(np.diag(dv) * self._molecule.mo.occs)
        dv = np.array(
            [dv[indices[0][0]][:, indices[1][0]], dv[indices[0][1]][:, indices[1][1]]]
        ).reshape([2 * self.M_block_size])
        dE = np.array([dE, -np.dot(dv[:, None].T, M_inv).dot(dv[:, None])[0, 0]])
        return dE
