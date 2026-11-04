import numpy as np
import matplotlib.pyplot as plt

class log_results:
    def __init__(self, out_dir_path):
        self.out_dir_path = out_dir_path

    def normal_log(self, k, f, err, converged, x):
        with open(f"{self.out_dir_path}/out.txt", "a") as f_log:
            f_log.write(f"     f = {f:.6e}, err = {err:.3e}\n")
            if converged:
                f_log.write(f"\nConverged!\n")
        n = int(np.sqrt(x.size(0)))
        Tmtx = x.view(n,n)
        Tmtx = np.array(Tmtx.detach().cpu())
        np.save(f"{self.out_dir_path}/translation_matrices/T_iter{k}.npy", Tmtx)
        # Save heatmap of Translation matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(Tmtx, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.title(f'Matrix shape: {Tmtx.shape}')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.savefig(f'{self.out_dir_path}/translation_matrix_visuals/T_iter{k}.png', dpi=300)
        
    def specific_log(self, message):
        with open(f"{self.out_dir_path}/out.txt", "a") as f_log:
            f_log.write(message+'\n')