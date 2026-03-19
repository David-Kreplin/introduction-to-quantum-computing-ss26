import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit.primitives import BackendSamplerV2
from qiskit.compiler import transpile


def sample_from_circuit_hardware(
    quantum_circuit: QuantumCircuit, num_shots: int
) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    # Initialize the sampler with 100 measurements (shots)
    backend = FakeWashingtonV2()
    return sample_from_circuit_backend(quantum_circuit, num_shots, backend)


def sample_from_circuit_backend(
    quantum_circuit: QuantumCircuit, num_shots: int, backend
) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    # Initialize the sampler with 100 measurements (shots)
    statevector_sampler = SamplerV2(mode=backend, options={"default_shots": num_shots})
    quantum_circuit_trans = transpile(
        quantum_circuit,
        backend=backend,
        basis_gates=backend.operation_names,
        coupling_map=backend.coupling_map,
    )

    # Execute the quantum circuit with measurements
    result = statevector_sampler.run([quantum_circuit_trans.reverse_bits()]).result()

    # Output the measurement results
    if hasattr(result[0].data, "c"):
        shots_dict = result[0].data.c.get_counts()
    elif hasattr(result[0].data, "meas"):
        shots_dict = result[0].data.meas.get_counts()
    else:
        raise ValueError("Only default register names are supported.")
    return shots_dict


def sample_from_circuit(quantum_circuit: QuantumCircuit, num_shots: int) -> dict:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the measurement results as a dictionary.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        dict: A dictionary with the measurement results.
    """

    # Initialize the sampler with 100 measurements (shots)
    backend = Aer.get_backend("aer_simulator")
    statevector_sampler = BackendSamplerV2(
        backend=backend, options={"default_shots": num_shots}
    )

    # Execute the quantum circuit with measurements
    result = statevector_sampler.run([quantum_circuit.reverse_bits()]).result()

    # Output the measurement results
    if hasattr(result[0].data, "c"):
        shots_dict = result[0].data.c.get_counts()
    elif hasattr(result[0].data, "meas"):
        shots_dict = result[0].data.meas.get_counts()
    else:
        raise ValueError("Only default register names are supported.")
    return shots_dict


def measure_to_probability(measurements: dict) -> dict:
    """
    Takes the measurement counts as a dictionary and converts them to probabilities.

    Args:
        measurements (dict): Dictionary with states as keys ans number of measurements
            as values

    Returns:
        Dictionary converted to measured probabilities
    """
    if not isinstance(measurements, dict):
        raise ValueError("Inputted measurements have to be a dictionary!")
    probs = {}
    total_counts = sum(measurements.values())
    for state in measurements.keys():
        probs[state] = measurements[state] / total_counts
    return probs


def sort_dict(shots_dict: dict):
    """
    Sorts a dictionary by its keys.

    Args:
        shots_dict (dict): The dictionary to be sorted.

    Returns:
        dict: The sorted dictionary.
    """

    return dict(sorted(shots_dict.items()))


def get_quasi_probs(quantum_circuit: QuantumCircuit, num_shots: int) -> np.ndarray:
    """
    Executes a quantum circuit with a specified number of measurements (shots)
    and returns the quasi-probabilities as an array.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
        num_shots (int): The number of measurements (shots) to be performed.

    Returns:
        np.ndarray: An array with the quasi-probabilities.
    """

    shots_dict = sample_from_circuit(quantum_circuit, num_shots)
    probs_array = convert_to_probs(shots_dict, quantum_circuit.num_qubits)
    return probs_array


def convert_to_probs(shots_dict: dict, num_qubits: int) -> np.ndarray:
    """
    Converts a dictionary with measurement results into an array of probabilities.

    Args:
        shots_dict (dict): A dictionary with the measurement results.
        num_qubits (int): The number of qubits in the quantum circuit.

    Returns:
        np.ndarray: An array with the probabilities.
    """

    num_states = 2**num_qubits
    probs_array = [0.0] * num_states

    for bitstring, count in shots_dict.items():
        index = int(bitstring, 2)
        probs_array[index] = count

    total_counts = sum(shots_dict.values())
    if total_counts > 0:
        probs_array = [count / total_counts for count in probs_array]

    return np.array(probs_array)


def measurements_to_probabilities(measurements: dict) -> dict:
    """
    Converts measurement counts to probabilities, both stored in dictionaries.

    Args:
        measurements (dict): A dictionary with measurement results as keys and their counts as values.

    Returns:
        dict: A dictionary with measurement results as keys and their probabilities as values.
    """
    probabilities = {}

    # Compute the total number of shots
    total_shots = 0
    for m in measurements.keys():
        total_shots += measurements[m]

    # Compute probabilities for each found basis state
    for m in measurements.keys():
        probabilities[m] = measurements[m] / total_shots

    return probabilities
