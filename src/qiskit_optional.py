def build_qiskit_qaoa_circuit(*args, **kwargs):
    """Build a placeholder-compatible Qiskit circuit only when Qiskit is installed."""
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise RuntimeError("Qiskit is optional and not installed in this environment.") from exc
    n_qubits = kwargs.get("n_qubits") or args[0]
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    return qc

