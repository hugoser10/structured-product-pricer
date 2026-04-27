import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize
from pricer.models.volatility.black_scholes import bs_call


class HestonModel:
    """
    dS = r*S*dt + sqrt(v)*S*dW^S
    dv = kappa*(theta - v)*dt + zeta*sqrt(v)*dW^v
    Pricing par Carr-Madan (FFT) ; Monte Carlo Euler pour exotiques.
    """

    def __init__(self, v0: float = 0.04, kappa: float = 1.5,
                 theta: float = 0.04, zeta: float = 0.3, rho: float = -0.7):
        self.v0, self.kappa, self.theta, self.zeta, self.rho = v0, kappa, theta, zeta, rho

    def _char_func(self, u: complex, S: float, r: float, T: float) -> complex:
        # Fonction caractéristique de log(S_T) sous Heston
        kappa, theta, zeta, rho, v0 = self.kappa, self.theta, self.zeta, self.rho, self.v0
        D1 = np.sqrt((kappa - zeta * rho * 1j * u) ** 2 + (u**2 + 1j * u) * zeta**2)
        g = (kappa - zeta * rho * 1j * u - D1) / (kappa - zeta * rho * 1j * u + D1)
        exp_D1_T = np.exp(-D1 * T)
        C_den = 1 - g * exp_D1_T
        C = (kappa - zeta * rho * 1j * u - D1) / zeta**2 * (T - 2 * np.log(C_den / (1 - g)))
        A = (r * 1j * u * T
             + kappa * theta / zeta**2 * ((kappa - zeta * rho * 1j * u - D1) * T
                                          - 2 * np.log(C_den / (1 - g))))
        
        return np.exp(A + C * v0 + 1j * u * np.log(S * np.exp(r * T)))

    def price_call_fourier(self, S: float, K: float, T: float, r: float,
                           n_points: int = 128) -> float:
        # Méthode de Carr-Madan
        if T <= 0:
            return max(S - K, 0.0)
        alpha = 1.5
        eta = 0.25
        N = n_points
        lam = 2 * np.pi / (N * eta)
        b = np.pi / eta
        k_arr = -b + lam * np.arange(N)
        v_arr = np.arange(N) * eta
        v_arr[0] = 1e-10

        psi = (np.exp(-r * T) * self._char_func(v_arr - (alpha + 1) * 1j, S, r, T)
               / (alpha**2 + alpha - v_arr**2 + 1j * (2 * alpha + 1) * v_arr))

        # Pondération de Simpson sur les nœuds FFT
        simpson = np.ones(N)
        simpson[0] = simpson[-1] = 1/3
        simpson[1:-1] = np.where(np.arange(1, N-1) % 2 == 1, 4/3, 2/3)

        fft_in = np.exp(1j * b * v_arr) * psi * eta * simpson
        call_values = np.real(np.exp(-alpha * k_arr) / np.pi * np.fft.fft(fft_in))

        # Interpolation linéaire au log-strike
        log_K = np.log(K)
        idx = np.clip(np.searchsorted(k_arr, log_K), 1, N - 2)
        w = (log_K - k_arr[idx - 1]) / (k_arr[idx] - k_arr[idx - 1])
        price = (1 - w) * call_values[idx - 1] + w * call_values[idx]
        return max(float(price), max(S - K * np.exp(-r * T), 0.0))

    def simulate_paths(self, S0: float, r: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None):
        # Schéma Euler à variance tronquée
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        S_paths = np.zeros((n_simulations, n_steps + 1))
        v_paths = np.zeros((n_simulations, n_steps + 1))
        S_paths[:, 0] = S0
        v_paths[:, 0] = self.v0

        u1 = np.random.normal(0, 1, (n_simulations, n_steps))
        u2 = np.random.normal(0, 1, (n_simulations, n_steps))
        z1 = u1

        # Cholesky 2x2 pour corréler dW^S et dW^v avec coefficient rho
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * u2

        for i in range(n_steps):
            v = v_paths[:, i]
            v_next = np.maximum(0, v + self.kappa * (self.theta - v) * dt
                                + self.zeta * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * z1[:, i])
            S_next = S_paths[:, i] * np.exp((r - 0.5 * v) * dt
                                            + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * z2[:, i])
            v_paths[:, i + 1] = v_next
            S_paths[:, i + 1] = S_next
        return S_paths, v_paths

    def implied_vol_from_heston(self, S, K, T, r) -> float:
        price = self.price_call_fourier(S, K, T, r)
        try:
            return brentq(lambda s: bs_call(S, K, T, r, s) - price, 1e-4, 5.0, xtol=1e-6)
        except Exception:
            return np.sqrt(self.v0)

    def calibrate(self, surface, r: float = 0.045):
        # Calibration en deux passes : DE global puis L-BFGS-B local
        S = surface.spot
        df = surface.get_surface_dataframe()

        # on restreint aux strikes proches de la monnaie pour éviter les wings illiquides
        df = df[(df["moneyness"] >= 0.7) & (df["moneyness"] <= 1.3)].copy()
        df_sample = df.sample(min(40, len(df)), random_state=42)
        atm_iv = surface.get_atm_vol(df["maturity"].mean())
        v0_init = max(atm_iv ** 2, 0.01)  # initialisation de v0 à la variance ATM observée

        def objective(params):
            v0, kappa, theta, zeta, rho = params
            if (v0 <= 0 or kappa <= 0 or theta <= 0 or zeta <= 0
                    or abs(rho) >= 0.999 or 2 * kappa * theta <= zeta ** 2):  # condition de Feller
                return 1e10
            tmp = HestonModel(v0, kappa, theta, zeta, rho)

            total = 0.0
            for _, row in df_sample.iterrows():
                try:
                    iv_h = tmp.implied_vol_from_heston(S, row["strike"], row["maturity"], r)
                    total += (iv_h - row["iv"] / 100.0) ** 2
                except Exception:
                    total += 0.1
            return total

        bounds = [
            (max(v0_init * 0.3, 0.001), min(v0_init * 3, 1.0)),
            (0.1, 8.0), (0.001, 0.5), (0.05, 1.5), (-0.95, 0.0),
        ]
        res_g = differential_evolution(objective, bounds, seed=42, maxiter=150,
                                        tol=1e-6, polish=False, workers=1)
        res = minimize(objective, res_g.x, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 200, "ftol": 1e-10})
        self.v0, self.kappa, self.theta, self.zeta, self.rho = res.x
        
        return self