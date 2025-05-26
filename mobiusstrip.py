import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.spatial.distance import euclidean

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        """
        Initialize the MobiusStrip object.

        Parameters:
        R : float - Radius from the center of the circle to the midline of the strip.
        w : float - Width of the strip.
        n : int   - Number of mesh points (resolution).
        """
        self.R = R
        self.w = w
        self.n = n

        
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, n),     # u ∈ [0, 2π]
            np.linspace(-w / 2, w / 2, n)     # v ∈ [−w/2, w/2]
        )

        
        self.x, self.y, self.z = self._compute_coordinates()

    def _compute_coordinates(self):
        """
        Generate (x, y, z) points for the Mobius strip using parametric equations.

        Returns:
        x, y, z : 2D numpy arrays representing the coordinates of the surface
        """
        u, v = self.u, self.v

       
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        return x, y, z

    def compute_surface_area(self):
        """
        Compute the surface area of the Mobius strip using numerical integration.

        Returns:
        float - Approximated surface area.
        """
        
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

       
        x_u, x_v = np.gradient(self.x, du, dv)
        y_u, y_v = np.gradient(self.y, du, dv)
        z_u, z_v = np.gradient(self.z, du, dv)

       
        cross_x = y_u * z_v - z_u * y_v
        cross_y = z_u * x_v - x_u * z_v
        cross_z = x_u * y_v - y_u * x_v

       
        dA = np.sqrt(cross_x**3 + cross_y**3 + cross_z**3)

        
        area = simps(simps(dA, dx=dv), dx=du)
        return area

    def compute_edge_length(self):
        """
        Compute the total edge length of the strip (top + bottom edges).

        Returns:
        float - Total edge length.
        """
        
        top_edge = np.stack((self.x[-1], self.y[-1], self.z[-1]), axis=1)
        bottom_edge = np.stack((self.x[0], self.y[0], self.z[0]), axis=1)

        
        def arc_length(curve):
            return sum(euclidean(curve[i], curve[i + 1]) for i in range(len(curve) - 1))

        
        return arc_length(top_edge) + arc_length(bottom_edge)

    def plot(self):
        """
        Render the Mobius strip in 3D using matplotlib.
        """
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')

        
        norm = plt.Normalize(self.z.min(), self.z.max())
        colors = plt.cm.viridis(norm(self.z))

        
        ax.plot_surface(self.x, self.y, self.z,
                        facecolors=colors, 
                        rstride=1, cstride=1, 
                        linewidth=0, antialiased=True)

       
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=30, azim=45)  

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
   
    mobius = MobiusStrip(R=1.0, w=0.3, n=300)

    
    print(f"Surface Area ≈ {mobius.compute_surface_area():.4f}")
    print(f"Edge Length  ≈ {mobius.compute_edge_length():.4f}")

    
    mobius.plot()
