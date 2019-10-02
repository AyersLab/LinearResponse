import numpy as np
from iodata import IOData
from iodata.orbitals import MolecularOrbitals
from gbasis.contractions import GeneralizedContractionShell
from grid.molgrid import MolGrid
from gbasis.evals.density import evaluate_density
from gbasis.evals.density import evaluate_density_gradient
from gbasis.evals.eval import evaluate_basis
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
import pylibxc


class K_matrices:
    """Class for the K matrices"""

    def __init__(self, basis, molecule, coord_type="spherical"):
        """Construct the K matrices and its related properties as defined in:
            Yang W, Cohen AJ, De Proft F, Geerlings P
            (2012) Analytical evaluation of Fukui functions and real-space linear
            response function. J Chem Phys 136:144110.

        Parameters
        ----------
        basis :  list/tuple of GeneralizedContractionShell
            Shells of generalized contractions.
        molecule : IOData class object for which the following attributes are defined :
            - mo.energies (energies of all the MO, alpha first, then beta)
            - mo.energiesa (energies of alpha MO)
            - mo.energiesb (energies of beta MO)
            - mo.coeffs (coefficients of all the MO on the AO basis, alpha first, then beta)
            - mo.coeffsa (coefficients of the alpha MO on the AO basis)
            - mo.coeffsb (coefficients of the beta MO on the AO basis)
            - mo.occs (occupation numbers of all the MO on the AO basis, alpha first, then beta)
            - mo.occsa (occupation numbers of the alpha orbitals)
            - mo.occsb (occupation numbers of the alpha orbitals)
            - mo.nbasis (number of basis functions)
        coord_type : {"cartesian", list/tuple of "cartesian" or "spherical", "spherical"}
            Types of the coordinate system for the contractions.
            If "cartesian", then all of the contractions are treated as Cartesian contractions.
            If "spherical", then all of the contractions are treated as spherical contractions.
            If list/tuple, then each entry must be a "cartesian" or "spherical" to specify the
            coordinate type of each `GeneralizedContractionShell` instance.
            Default value is "spherical"

        Raises
        ------
        TypeError
            If 'basis' is not a list
            If 'basis' is not a list of 'GeneralizedContractionShell' instances
            If 'molecule' is not a 'IOData' instance
            If 'molecule.mo' is not a 'MolecularOrbitals' instance
        ValueError
            If 'coordtype' is not 'cartesian', 'spherical' or a list/tuple of 'cartesian' or 'spherical'
        """
        if not isinstance(basis, list):
            raise TypeError(
                """'basis' must be a list of 'GeneralizedContractionShell' instances"""
            )
        for i in range(len(basis)):
            if not isinstance(basis[i], GeneralizedContractionShell):
                raise TypeError(
                    """'basis' must be a list of 'GeneralizedContractionShell' instances"""
                )
        if not isinstance(molecule, IOData):
            raise TypeError("""'molecule' must be a 'IOData' instance""")
        if not isinstance(molecule.mo, MolecularOrbitals):
            raise TypeError("""'molecule.mo' must be a 'MolecularOrbitals' instance""")
        if isinstance(coord_type, str):
            if not coord_type in ["cartesian", "spherical"]:
                raise ValueError(
                    """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
                )
        elif isinstance(coord_type, tuple) or isinstance(coord_type, list):
            for i in range(len(coord_type)):
                if not coord_type[i] in ["cartesian", "spherical"]:
                    raise ValueError(
                        """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
                    )
        else:
            raise TypeError(
                """'coordtype' must be 'cartesian', 'spherical' or list/tuple of 'cartesian' or 'spherical'"""
            )
        self._basis = basis
        self._molecule = molecule
        self._coord_type = coord_type
        self._two_electron_integrals = None

    @property
    def two_electron_integrals(self):
        """Two electron integrals array

        Calculate and store the two electron integrals so that they are computed only once """
        if not isinstance(self._two_electron_integrals, np.ndarray):
            self._two_electron_integrals = electron_repulsion_integral(
                self._basis,
                transform=self._molecule.mo.coeffs.T,
                coord_type=self._coord_type,
                notation="chemist",
            )
        return self._two_electron_integrals

    def K_indices(self, shape="square", index=None):
        """Generate the lists of indices for construction of the coupling matrix K

        Parameters
        ----------
        shape: str; 'square', 'line' or 'point', default is 'square'
            If shape == 'square', generate the lists of indices to generate (K)ias,jbt
                Useful to generate the M matrices
            If shape ==  'line' ,
                generate the lists of indices to generate (K)ffs,jbt
                Useful to generate the fukui matrices and hardness
            If shape == 'point'
                generate the lists of indices to generate (K)ffs,ffs
                Useful to generate the hardness
        index: None or list of two list of int, default is None
            Necessary if shape == 'line' or shape ==  'point'
            If index is a list of two list of integers, it must be like [ l1, l2] with l1 in |[0; molecule.mo.nbasis|[
            and l2 in |[molecule.mo.nbasis, 2 * molecule.mo.nbasis|[, (l1 : alpha indices, l2 : beta indices)
            len(l1) + len(l2) must be equal to 1 if shape is 'line'
            len(l1) + len(l2) must be equal to 2 if shape is 'point'

        Return
        ------
        K_indices : tuple of lists
            If shape == 'square':
            ([alpha_occupied indices, beta_occupied indices],[alpha_virtual_indices, beta_virtual_indices])
            If shape == 'line' or shape == 'point,
            ([alpha_occupied indices, beta_occupied indices],[alpha_virtual_indices, beta_virtual_indices], index)

        Raises
        ------
        TypeError:
            If 'index' is not a list of two lists
        ValueError:
            ValueError:
            If shape is not 'square' or 'line' or 'point'
            If len(index[0]) + len(index[1]) != 1/2 depending on shape
            If index[0][0] is not in list(range(molecule.mo.nbasis)) or index[1][0] not in list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))
            """
        if not isinstance(shape, str):
            raise TypeError("""'shape' must be 'square' or 'line' or 'point' """)
        if not shape in ["square", "line", "point"]:
            raise ValueError("""'shape' must be 'square' or 'line' or 'point' """)
        if index is not None:
            if not isinstance(index, list):
                raise TypeError(
                    """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
                )
            if len(index) != 2:
                raise TypeError(
                    """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
                )
            if not isinstance(index[0], list):
                raise TypeError(
                    """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
                )
            if not isinstance(index[1], list):
                raise TypeError(
                    """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis))"""
                )
            if shape == "line":
                if len(index[0]) + len(index[1]) != 1:
                    raise ValueError(
                        """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
                    )
                else:
                    if len(index[0]) == 1:
                        if index[0][0] not in list(range(self._molecule.mo.nbasis)):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
                            )
                    else:
                        if index[1][0] not in list(
                            range(
                                self._molecule.mo.nbasis, 2 * self._molecule.mo.nbasis
                            )
                        ):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 1 as 'shape' == 'line'"""
                            )
            else:
                if len(index[0]) + len(index[1]) != 2:
                    raise ValueError(
                        """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                    )
                else:
                    if len(index[0]) == 1:
                        if index[0][0] not in list(range(self._molecule.mo.nbasis)):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
                        if index[1][0] not in list(
                            range(
                                self._molecule.mo.nbasis, 2 * self._molecule.mo.nbasis
                            )
                        ):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
                    if len(index[0]) == 0:
                        if index[1][0] not in list(
                            range(
                                self._molecule.mo.nbasis, 2 * self._molecule.mo.nbasis
                            )
                        ):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
                        if index[1][1] not in list(
                            range(
                                self._molecule.mo.nbasis, 2 * self._molecule.mo.nbasis
                            )
                        ):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
                    if len(index[0]) == 2:
                        if index[0][0] not in list(range(self._molecule.mo.nbasis)):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
                        if index[0][1] not in list(range(self._molecule.mo.nbasis)):
                            raise ValueError(
                                """'index' must be a list of two list of integers like [l1, l2] with l1 a slice of list(range(molecule.mo.nbasis)), l2 a slice of list(range(molecule.mo.nbasis, 2*molecule.mo.nbasis)) and with len(l1) + len(l2) == 2 as 'shape' == 'point'"""
                            )
        occupied_ind = [
            np.where(self._molecule.mo.occsa == 1.0)[0].tolist(),
            (
                np.where(self._molecule.mo.occsb == 1.0)[0] + self._molecule.mo.nbasis
            ).tolist(),
        ]
        virtual_ind = [
            np.where(self._molecule.mo.occsa == 0.0)[0].tolist(),
            (
                np.where(self._molecule.mo.occsb == 0.0)[0] + self._molecule.mo.nbasis
            ).tolist(),
        ]
        if shape == "line" or shape == "point":
            K_indices = (occupied_ind, virtual_ind, index)
        else:
            K_indices = (occupied_ind, virtual_ind)
        return K_indices

    def K_shape(self, shape="square", index=None):
        """Calculate the size of the K matrix

        Parameters
        ----------
        shape: str; 'square', 'line' or 'point', default is 'square'
            If shape == 'square', generate the shape of the square K matrix
            If shape ==  'line' , generate the shape of the line K matrix
                Useful to generate the fukui matrices and hardness
            If shape == 'point', generate the shape of the point K matrix
        index: None or list of two list of int, default is None
            Necessary if shape == 'line' or shape ==  'point'
            If index is a list of two list of integers, it must be like [ l1, l2] with l1 in |[0; molecule.mo.nbasis|[
            and l2 in |[molecule.mo.nbasis, 2 * molecule.mo.nbasis|[, (l1 : alpha indices, l2 : beta indices)
            len(l1) + len(l2) must be equal to 1 if shape is 'line'
            len(l1) + len(l2) must be equal to 2 if shape is 'point'

        Return
        ------
        K_size : list of two int"""
        indices = self.K_indices(shape=shape, index=index)
        if shape == "square":
            K_shape = [
                len(indices[0][0]) * len(indices[1][0]),
                len(indices[0][0]) * len(indices[1][0]),
            ]
        elif shape == "line":
            K_shape = [
                len(indices[2][0]) + len(indices[2][1]),
                len(indices[0][0]) * len(indices[1][0]),
            ]
        else:
            K_shape = [1, 1]
        return K_shape

    def K_coulomb(self, Type=1, shape="square", index=None):
        """Calculate the two electron integral part of the K matrix

        Parameters
        ----------
        Type : int,
            1 or 2, in cassee of real molecular orbitals, type1 = type2
        shape: str; 'square', 'line' or 'point', default is 'square'
            If shape == 'square', generate sq_(K_coulomb)
                Useful to generate the M matrices
            If shape ==  'line' ,
                generate li_(K_coulomb)
                Useful to generate the fukui matrices and hardness
            If shape == 'point'
                generate pt_(K_coulomb)
                Useful to generate the hardness
        index: None or list of two list of int, default is None
            Necessary if shape == 'line' or shape ==  'point'
            If index is a list of two list of integers, it must be like [ l1, l2] with l1 in |[0; molecule.mo.nbasis|[
            and l2 in |[molecule.mo.nbasis, 2 * molecule.mo.nbasis|[, (l1 : alpha indices, l2 : beta indices)
            len(l1) + len(l2) must be equal to 1 if shape is 'line'
            len(l1) + len(l2) must be equal to 2 if shape is 'point'

        Return
        ------
        K_coulomb: ndarray
            shape = [aa_block, ab_block, ba_block,bb_block] if shape ==  'square' or 'line'
            shape = [1, 1] if shape == 'point'

        Raises
        ------
        TypeError
            If 'Type'  is not an int
        ValueError
            If 'Type' is not in [1,2]"""
        if not isinstance(Type, int):
            raise TypeError("""'type' must be 1 or 2""")
        if not Type in [1, 2]:
            raise ValueError("""'type' must be 1 or 2""")
        indices = self.K_indices(shape=shape, index=index)
        two_electron_int = self.two_electron_integrals
        if Type == 1:
            p, q = 1, 0
        else:
            p, q = 0, 1

        if shape == "square":
            two_electron_int = two_electron_int[indices[0][0] + indices[0][1]][
                :, indices[1][0] + indices[1][1]
            ][:, :, indices[p][0] + indices[p][1]][
                :, :, :, indices[q][0] + indices[q][1]
            ]
            aa_block = two_electron_int[: len(indices[0][0])][:, : len(indices[1][0])][
                :, :, : len(indices[p][0])
            ][:, :, :, : len(indices[q][0])].reshape(
                [
                    len(indices[0][0]) * len(indices[1][0]),
                    len(indices[p][0]) * len(indices[q][0]),
                ]
            )
            ab_block = two_electron_int[: len(indices[0][0])][:, : len(indices[1][0])][
                :, :, len(indices[p][0]) :
            ][:, :, :, len(indices[q][0]) :].reshape(
                [
                    len(indices[0][0]) * len(indices[1][0]),
                    len(indices[p][1]) * len(indices[q][1]),
                ]
            )
            ba_block = two_electron_int[: len(indices[0][0])][:, : len(indices[1][0])][
                :, :, len(indices[p][0]) :
            ][:, :, :, len(indices[q][0]) :].reshape(
                [
                    len(indices[0][1]) * len(indices[1][1]),
                    len(indices[p][0]) * len(indices[q][0]),
                ]
            )
            bb_block = two_electron_int[len(indices[0][0]) :][:, len(indices[1][0]) :][
                :, :, len(indices[p][0]) :
            ][:, :, :, len(indices[q][0]) :].reshape(
                [
                    len(indices[0][1]) * len(indices[1][1]),
                    len(indices[p][1]) * len(indices[q][1]),
                ]
            )
            K_coulomb = np.array([aa_block, ab_block, ba_block, bb_block])
            return K_coulomb
        elif shape == "line":
            two_electron_int = two_electron_int[indices[2][0] + indices[2][1]][
                :, indices[2][0] + indices[2][1]
            ][:, :, indices[p][0] + indices[p][1]][
                :, :, :, indices[q][0] + indices[q][1]
            ]
            aa_block = (
                np.diagonal(
                    two_electron_int[: len(indices[2][0])][:, : len(indices[2][0])][
                        :, :, : len(indices[p][0])
                    ][:, :, :, : len(indices[q][0])]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][0]), len(indices[p][0]) * len(indices[q][0])])
            )
            ab_block = (
                np.diagonal(
                    two_electron_int[: len(indices[2][0])][:, : len(indices[2][0])][
                        :, :, : len(indices[p][1])
                    ][:, :, :, : len(indices[q][1])]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][0]), len(indices[p][1]) * len(indices[q][1])])
            )
            ba_block = (
                np.diagonal(
                    two_electron_int[len(indices[2][0]) :][:, len(indices[2][0]) :][
                        :, :, len(indices[p][0]) :
                    ][:, :, :, len(indices[q][0]) :]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][1]), len(indices[p][0]) * len(indices[q][0])])
            )
            bb_block = (
                np.diagonal(
                    two_electron_int[len(indices[2][0]) :][:, len(indices[2][0]) :][
                        :, :, len(indices[p][0]) :
                    ][:, :, :, len(indices[q][0]) :]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][1]), len(indices[p][1]) * len(indices[q][1])])
            )
            K_coulomb = np.array([aa_block, ab_block, ba_block, bb_block])
            return K_coulomb
        else:
            l = indices[2][0] + indices[2][1]
            return np.array([[two_electron_int[l[0], l[0], l[1], l[1]]]])

    def K_fxc_HF(self, Type=1, shape="square", index=None):
        """Calculate the exchange-correlation part of the K matrix

        Parameters
        ----------
        Type : int,
            1 or 2, in cassee of real molecular orbitals, type1 = type2
        shape: str; 'square', 'line' or 'point', default is 'square'
            If shape == 'square', generate sq_(K_coulomb)
                Useful to generate the M matrices
            If shape ==  'line' ,
                generate li_(K_coulomb)
                Useful to generate the fukui matrices and hardness
            If shape == 'point'
                generate pt_(K_coulomb)
                Useful to generate the hardness
        index: None or list of two list of int, default is None
            Necessary if shape == 'line' or shape ==  'point'
            If index is a list of two list of integers, it must be like [ l1, l2] with l1 in |[0; molecule.mo.nbasis|[
            and l2 in |[molecule.mo.nbasis, 2 * molecule.mo.nbasis|[, (l1 : alpha indices, l2 : beta indices)
            len(l1) + len(l2) must be equal to 1 if shape is 'line'
            len(l1) + len(l2) must be equal to 2 if shape is 'point'

        Return
        ------
        K_fxc_HF: ndarray
            [aa_block, ab_block, ba_block,bb_block] if shape ==  'square' or 'line'

        Raises
        ------
        TypeError
            If 'Type'  is not an int
        ValueError
            If 'Type' is not in [1,2]"""
        if not isinstance(Type, int):
            raise TypeError("""'Type' must be 1 or 2""")
        if not Type in [1, 2]:
            raise ValueError("""'type' must be 1 or 2""")
        indices = self.K_indices(shape=shape, index=index)
        if Type == 1:
            p, q = 1, 0
        else:
            p, q = 0, 1

        two_electron_int = self.two_electron_integrals

        if shape == "square":
            two_electron_int = two_electron_int[indices[0][0] + indices[0][1]][
                :, indices[1][0] + indices[1][1]
            ][:, :, indices[p][0] + indices[p][1]][
                :, :, :, indices[q][0] + indices[q][1]
            ]
            aa_block = two_electron_int[: len(indices[0][0])][:, : len(indices[1][0])][
                :, :, : len(indices[p][0])
            ][:, :, :, : len(indices[q][0])].reshape(
                [
                    len(indices[0][0]) * len(indices[1][0]),
                    len(indices[p][0]) * len(indices[q][0]),
                ]
            )
            ab_block = np.zeros(
                [
                    len(indices[0][0]) * len(indices[1][0]),
                    len(indices[p][1]) * len(indices[q][1]),
                ]
            )
            ba_block = np.zeros(
                [
                    len(indices[0][1]) * len(indices[1][1]),
                    len(indices[p][0]) * len(indices[q][0]),
                ]
            )
            bb_block = two_electron_int[len(indices[0][0]) :][:, len(indices[1][0]) :][
                :, :, len(indices[p][0]) :
            ][:, :, :, len(indices[q][0]) :].reshape(
                [
                    len(indices[0][1]) * len(indices[1][1]),
                    len(indices[p][1]) * len(indices[q][1]),
                ]
            )
            K_fxc_HF = (-1) * np.array([aa_block, ab_block, ba_block, bb_block])
            return K_fxc_HF
        elif shape == "line":
            two_electron_int = two_electron_int[indices[2][0] + indices[2][1]][
                :, indices[2][0] + indices[2][1]
            ][:, :, indices[p][0] + indices[p][1]][
                :, :, :, indices[q][0] + indices[q][1]
            ]
            aa_block = (
                np.diagonal(
                    two_electron_int[: len(indices[2][0])][:, : len(indices[2][0])][
                        :, :, : len(indices[p][0])
                    ][:, :, :, : len(indices[q][0])]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][0]), len(indices[p][0]) * len(indices[q][0])])
            )
            ab_block = np.zeros(
                [len(indices[2][0]), len(indices[p][1]) * len(indices[q][1])]
            )
            ba_block = np.zeros(
                [len(indices[2][1]), len(indices[p][0]) * len(indices[q][0])]
            )
            bb_block = (
                np.diagonal(
                    two_electron_int[len(indices[2][0]) :][:, len(indices[2][0]) :][
                        :, :, len(indices[p][0]) :
                    ][:, :, :, len(indices[q][0]) :]
                )
                .transpose(2, 0, 1)
                .reshape([len(indices[2][1]), len(indices[p][1]) * len(indices[q][1])])
            )
            K_fxc_HF = (-1) * np.array([aa_block, ab_block, ba_block, bb_block])
            return K_fxc_HF
        else:
            l = indices[2][0] + indices[2][1]
            if len(indices[2][0]) == 2 or len(indices[2][1]) == 2:
                return np.array([[-two_electron_int[l[0], l[0], l[1], l[1]]]])
            else:
                return np.array([[0.0]])

    def K_fxc_DFT(
        self, molgrid, Type=1, XC_functional="lda_x", shape="square", index=None
    ):
        """Calculate the exchange-correlation part of the coupling matrix K

        Parameters
        ----------
        molgrid : MolGrid class object (from grid package)  suitable for numerical integration
            of any real space function related to the molecule (such as the density)
        Type : int,
            1 or 2, in cassee of real molecular orbitals, type1 = type2
        XC_functionl : str
            Code name of the exchange correlation functional as given on the
            page: https://tddft.org/programs/libxc/functionals/
        shape: str; 'square', 'rectangle', 'line' or 'point', default is 'square'
            If shape == 'square', generate (K_fxc_DFT)ias,jbt
                Useful to generate the M matrices
            If shape ==  'line' ,
                generate (K_fxc_DFT)ffs,jbt
                Useful to generate the fukui matrices and hardness
            If shape == 'point'
                generate (K_fxc_DFT)ffs,ffs
                Useful to generate the hardness
        index: None or list of two list of int, default is None
            Necessary if shape == 'line' or shape ==  'point'
            If index is a list of two list of integers, it must be like [ l1, l2] with l1 in |[0; molecule.mo.nbasis|[
            and l2 in |[molecule.mo.nbasis, 2 * molecule.mo.nbasis|[, (l1 : alpha indices, l2 : beta indices)
            len(l1) + len(l2) must be equal to 1 if shape is 'line'
            len(l1) + len(l2) must be equal to 2 if shape is 'point'

        Return
        ------
        K_fxc_DFT: ndarray
            [aa_block, ab_block, ba_block, bb_block] if shape ==  'square' or 'line'

        Raises
        ------
        TypeError
            If 'molgrid' is not a 'MolGrid' instance'
            If 'Type' is not an int
            XC_functional is not a str
        ValueError
            If 'Type' is not in [1,2]
            If XC_functional is bot supported by pylibxc (see pylibxc.util.xc_available_functional_names() or https://tddft.org/programs/libxc/functionals/ for the function code spelling)
            IF XC_functional is not a LDA or a GGA"""
        if not isinstance(molgrid, MolGrid):
            raise TypeError("""'molgrid' must be a 'MolGrid' instance""")
        if not isinstance(Type, int):
            raise TypeError("""'type' must be 1 or 2""")
        if not Type in [1, 2]:
            raise ValueError("""'type' must be 1 or 2""")
        if not isinstance(XC_functional, str):
            raise TypeError("""'XC_functional' must be a str""")
        if not XC_functional in pylibxc.util.xc_available_functional_names():
            raise ValueError(
                """"Not suported functionnal, see pylibxc.util.xc_available_functional_names() or the webpage: https://tddft.org/programs/libxc/functionals/"""
            )
        xc_function = pylibxc.LibXCFunctional(XC_functional, "polarized")
        if not xc_function.get_family() in [1, 2]:
            raise ValueError(
                """"Not suported functionnal, only the LDA ans GGAs are supported"""
            )

        # Defining the array of values of fxc(r) suitable for the numerical integration
        a_dm = np.dot(
            self._molecule.mo.coeffsa * self._molecule.mo.occsa,
            self._molecule.mo.coeffsa.T.conj(),
        )
        b_dm = np.dot(
            self._molecule.mo.coeffsb * self._molecule.mo.occsb,
            self._molecule.mo.coeffsb.T.conj(),
        )
        a_density_values = evaluate_density(
            a_dm, self._basis, molgrid.points, coord_type=self._coord_type
        )
        b_density_values = evaluate_density(
            b_dm, self._basis, molgrid.points, coord_type=self._coord_type
        )
        xc_function = pylibxc.LibXCFunctional(XC_functional, "polarized")
        inp = {"rho": np.array([a_density_values, b_density_values])}
        inp["rho"] = inp["rho"].transpose().reshape(len(inp["rho"][0]) * 2)
        if xc_function.get_family() == 2:
            a_gradient_density = evaluate_density_gradient(
                a_dm, self._basis, molgrid.points, coord_type=self._coord_type
            )
            b_gradient_density = evaluate_density_gradient(
                b_dm, self._basis, molgrid.points, coord_type=self._coord_type
            )
            inp["sigma"] = np.array(
                [
                    np.sum(a_gradient_density * a_gradient_density, axis=1),
                    np.sum(a_gradient_density * b_gradient_density, axis=1),
                    np.sum(b_gradient_density * b_gradient_density, axis=1),
                ]
            )
            inp["sigma"] = inp["sigma"].transpose().reshape(len(inp["sigma"][0]) * 3)
        f_xc_values = (xc_function.compute(inp, None, False, False, True, False))[
            "v2rho2"
        ]
        f_xc_values = np.array(
            [
                f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 0],
                f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 1],
                f_xc_values.reshape(len(f_xc_values[0]), 3)[:, 2],
            ]
        )
        if xc_function.get_family() == 2:
            del (a_gradient_density, b_gradient_density)
        del (a_dm, b_dm, a_density_values, b_density_values, xc_function, inp)

        MO_basis_func_val = evaluate_basis(
            self._basis,
            molgrid.points,
            transform=self._molecule.mo.coeffs.T,
            coord_type=self._coord_type,
        )
        indices = self.K_indices(shape=shape, index=index)
        # Defining the arrays of values of all the 4 MO products suitable for the numerical integration
        if shape == "square" or shape == "line":
            a_occup_val = MO_basis_func_val[indices[0][0]]
            a_virt_val = MO_basis_func_val[indices[1][0]]
            kl_a = (a_occup_val[:, None].conj() * a_virt_val).reshape(
                [a_occup_val.shape[0] * a_virt_val.shape[0], a_virt_val.shape[1]]
            )
            del (a_occup_val, a_virt_val)
            b_occup_val = MO_basis_func_val[indices[0][1]]
            b_virt_val = MO_basis_func_val[indices[1][1]]
            kl_b = (b_occup_val[:, None].conj() * b_virt_val).reshape(
                [b_occup_val.shape[0] * b_virt_val.shape[0], b_virt_val.shape[1]]
            )
            del (b_occup_val, b_virt_val)
            klt = np.block([kl_a.T, kl_b.T]).T
            del kl_b
            if shape == "line":
                a_i_values = MO_basis_func_val[indices[2][0]]
                b_j_values = MO_basis_func_val[indices[2][1]]
                ij_a = a_i_values.conj() * a_i_values
                ij_b = b_j_values.conj() * b_j_values
                ijs = np.block([ij_a.T, ij_b.T]).T
                del (a_i_values, b_j_values, ij_b)
            else:
                ijs = klt
                ij_a = kl_a
            del MO_basis_func_val
            if type == 2:
                K_fxc_DFT = ijs[:, None] * klt
            else:
                K_fxc_DFT = ijs[:, None] * klt.conj()
            # generate the alpha-alpha block
            K_fxc_DFT[: ij_a.shape[0], : kl_a.shape[0]] = (
                K_fxc_DFT[: ij_a.shape[0], : kl_a.shape[0]]
                * f_xc_values[0]
                * molgrid.weights
            )
            # generate the alpha-beta block
            K_fxc_DFT[ij_a.shape[0] :, : kl_a.shape[0]] = (
                K_fxc_DFT[ij_a.shape[0] :, : kl_a.shape[0]]
                * f_xc_values[1]
                * molgrid.weights
            )
            # generate the beta-alpha block
            K_fxc_DFT[: ij_a.shape[0], kl_a.shape[0] :] = (
                K_fxc_DFT[: ij_a.shape[0], kl_a.shape[0] :]
                * f_xc_values[1]
                * molgrid.weights
            )
            # generate th beta-beta block
            K_fxc_DFT[ij_a.shape[0] :, kl_a.shape[0] :] = (
                K_fxc_DFT[ij_a.shape[0] :, kl_a.shape[0] :]
                * f_xc_values[2]
                * molgrid.weights
            )
            # integrate
            K_fxc_DFT = np.sum(K_fxc_DFT / 3, axis=2)
            K_fxc_DFt = np.array(
                [
                    K_fxc_DFT[: ij_a.shape[0], : kl_a.shape[0]],
                    K_fxc_DFT[ij_a.shape[0] :, : kl_a.shape[0]],
                    K_fxc_DFT[: ij_a.shape[0], kl_a.shape[0] :],
                    K_fxc_DFT[ij_a.shape[0] :, kl_a.shape[0] :],
                ]
            )
            return K_fxc_DFt
        else:
            l = indices[2][0] + indices[2][1]
            i_values = MO_basis_func_val[l[0]]
            k_values = MO_basis_func_val[l[1]]
            K_fxc_DFT = i_values * i_values.conj() * k_values * k_values.conj()
            if len(indices[2][0]) == 0:
                return np.sum(K_fxc_DFT * f_xc_values[2] * molgrid.weights / 3)
            if len(indices[2][0]) == 1:
                return np.sum(K_fxc_DFT * f_xc_values[1] * molgrid.weights / 3)
            else:
                return np.sum(K_fxc_DFT * f_xc_values[0] * molgrid.weights / 3)
