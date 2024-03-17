import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

try:
    # Define constants
    c = 3e8  # Speed of light in meters/second
   
    # Define simulation parameters
    resolution = 8
    cell_size = mp.Vector3(10, 10, 0)
    pml_layers = [mp.PML(1.0)]
   
    # Define materials
    materials = [mp.Medium(epsilon=eps) for eps in [12, 6, 3, 9]]
   
    # Define source
    source_frequency = 0.5  # This is in Meep's normalized units
    source = [mp.Source(mp.ContinuousSource(frequency=source_frequency), component=mp.Ez, center=mp.Vector3(0, 0, -5))]

    all_dft_data = []
    all_field_distributions = []
   
    # Optimization settings
    N = 4
    bounds_thickness = (0.5, 5.0)
    bounds_material = (0, len(materials) - 1)
    bounds = [bounds_thickness if i % 2 == 0 else bounds_material for i in range(2 * N)]
   
    # Define the characteristic length scale (e.g., size of the unit cell)
    length_scale = 1e-6  # in meters, example for a structure with features in the micrometer range
   
    # Frequency settings for DFT (in physical units like THz)
    fmin_physical = 0.1e12  # 0.1 THz
    fmax_physical = 2.0e12  # 2 THz
   
    # Convert physical frequencies to Meep's normalized frequency units
    fmin_meep = fmin_physical * length_scale / c
    fmax_meep = fmax_physical * length_scale / c
   
    # Number of frequency points
    nfreq = 50
   
    # Generate frequency array for DFT in Meep's units
    frequencies_meep = np.linspace(fmin_meep, fmax_meep, nfreq)
    global frequencies_hz
    # Convert frequencies from Meep units to Hz for plotting and analysis
    frequencies_hz = frequencies_meep * c / length_scale
   
    # Define the geometry creation function
    def create_geometry(params):
        geometry = []
        z_center = 0
        for i in range(0, len(params), 2):
            thickness = params[i]
            material_idx = int(np.round(params[i + 1])) % len(materials)
            material = materials[material_idx]
            block = mp.Block(material=material, size=mp.Vector3(10, 10, thickness), center=mp.Vector3(0, 0, z_center))
            geometry.append(block)
            z_center += thickness
        return geometry
   
    # Define the simulation function
    def run_simulation(params):
        geometry = create_geometry(params)
        sim = mp.Simulation(cell_size=cell_size, geometry=geometry, sources=source, boundary_layers=pml_layers, resolution=resolution)
        dft_obj = sim.add_dft_fields([mp.Ez], fmin_meep, fmax_meep, nfreq, where=mp.Volume(center=mp.Vector3(), size=cell_size))
        sim.run(until=200)
        return sim, dft_obj
       
    def objective(params):
        global all_dft_data, all_field_distributions
        sim, dft_obj = run_simulation(params)
        ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
        all_field_distributions.append(ez_data)
       
        dft_data = np.zeros(nfreq, dtype=np.complex128)
        for i in range(nfreq):
            dft_data[i] = np.mean(sim.get_dft_array(dft_obj, mp.Ez, i))
        all_dft_data.append(dft_data)
        abs_dft = np.abs(dft_data)
        peaks, _ = find_peaks(abs_dft, height=0.01)
       
        if len(peaks) == 0:
            return -np.inf
       
        total_peaks = len(peaks)
        bandwidth = frequencies_hz[peaks[-1]] - frequencies_hz[peaks[0]]
        material_diversity = len(set(params[1::2]))
        diversity_penalty = (N - material_diversity) * 0.1
        score = bandwidth * total_peaks - diversity_penalty
       
        return -score
   
    def global_optimization():
        result = differential_evolution(objective, bounds)
        if result.success:
            print("Optimized Parameters:", result.x)
            print("Objective Value:", -result.fun)
            return result.x
        else:
            print("Optimization failed:", result.message)
            return None
        # Plotting functions

    '''
    def plot_field_distribution(sim, dft_obj, frequencies_hz):
        ez_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
        plt.figure(figsize=(8, 6))
       
        plt.imshow(np.abs(ez_data.transpose()), interpolation='spline36', cmap='viridis')
        plt.colorbar()
        plt.title('Ez Field Distribution')
        plt.savefig("ez_field_distribution.png")
       
   
        dft_data = np.zeros(nfreq, dtype=np.complex128)
        for i in range(nfreq):
            dft_data[i] = np.mean(sim.get_dft_array(dft_obj, mp.Ez, i))
       
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies_hz, np.abs(dft_data))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT of Ez (Arbitrary Units)")
        plt.title("DFT Results Across the Frequency Domain")
        plt.grid(True)
        plt.savefig("dft_results.png")


    def plot_dft_results(sim, dft_obj, frequencies_hz):
        dft_data = np.zeros(nfreq, dtype=np.complex128)
        for i in range(nfreq):
            dft_data[i] = np.mean(sim.get_dft_array(dft_obj, mp.Ez, i))
       
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies_hz, np.abs(dft_data))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT of Ez (Arbitrary Units)")
        plt.title("DFT Results Across the Frequency Domain")
        plt.grid(True)
        plt.savefig("dft_results.png")

    '''
    def finalize_plots():
        global all_dft_data, all_field_distributions, frequencies_hz
        # Find the index of the best simulation based on some criterion, e.g., the highest peak
        # For simplicity, let's assume the last simulation was the best
        best_index = len(all_dft_data) - 1
       
        # Plot the field distribution for the best simulation
        ez_data = all_field_distributions[best_index]
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(ez_data.transpose()), interpolation='spline36', cmap='viridis')
        plt.colorbar()
        plt.title('Ez Field Distribution of Best Simulation')
        plt.savefig("best_ez_field_distribution.png")
       
        # Plot the DFT results for the best simulation
        dft_data = all_dft_data[best_index]
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies_hz, np.abs(dft_data))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT of Ez (Arbitrary Units)")
        plt.title("DFT Results Across the Frequency Domain of Best Simulation")
        plt.grid(True)
        plt.savefig("best_dft_results.png")


    # Running the global optimization
    best_params = global_optimization()
       
    # Check if we have the optimized parameters and run the final simulation
    if best_params is not None:
        sim, dft_obj = run_simulation(best_params)
        #plot_dft_results(sim, dft_obj, frequencies_hz)

except:
    print("Optimization was interrupted. Finalizing plots with the current best results.")
    finalize_plots()


finally:
    finalize_plots()
    #plot_dft_results(sim, dft_obj, frequencies_hz)
    #plot_field_distribution(sim, dft_obj, frequencies_hz)