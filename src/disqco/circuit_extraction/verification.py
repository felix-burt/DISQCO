import matplotlib.pyplot as plt
from qiskit import transpile
import numpy as np

def run_sampler(circuit, shots=4096):
    from qiskit_aer.primitives import SamplerV2
    sampler = SamplerV2()
    num_qubits = circuit.num_qubits
    dec_circuit = circuit.copy()
    dec_circuit = transpile(dec_circuit, basis_gates=['u', 'cp', 'EPR'])
    dec_circuit = dec_circuit.decompose()
    if num_qubits < 12:

        job = sampler.run([dec_circuit], shots=shots)
        job_result = job.result()
        data = job_result[0].data
    else:
        print("Too many qubits")
        data = None
    return data

def plot(data):
    from qiskit.visualization import plot_histogram
    if data is None:
        print("No data to plot")
        return
    if 'result' in data:
        info = data['result']
    else:
        info = data['meas']

    counts_base = info.get_counts()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_histogram(counts_base, bar_labels=False, ax=ax)
    ax.set_xticks([])

def get_fidelity(data1, data2, shots):
    if data1 is None or data2 is None:
        print("No data to compare")
        return None
    if 'result' in data1:
        info1 = data1['result']
    else:
        info1 = data1['meas']

    if 'result' in data2:
        info2 = data2['result']
    else:
        info2 = data2['meas']
    
    counts1 = info1.get_counts()
    counts2 = info2.get_counts()
    for key in counts1:
        digits = len(key)
        break
    norm = 0    
    max_string = '1'*digits
    integer = int(max_string, 2)
    for i in range(integer+1):
        binary = bin(i)
        binary = binary[2:]
        binary = '0'*(digits-len(binary)) + binary
        if binary in counts1:
            counts1_val = counts1[binary]/shots
        else:
            counts1_val = 0
        if binary in counts2:
            counts2_val = counts2[binary]/shots
        else:
            counts2_val = 0
        norm += np.abs(counts1_val - counts2_val)
    return 1 - norm**2