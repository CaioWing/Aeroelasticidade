import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp_special
import sympy as sp

def Ck(k):
    """
    Função de Theodorsen
    """
    y = (sp_special.hankel2(1, k)) / ((sp_special.hankel2(1, k)) + (1j * (sp_special.hankel2(0, k))))
    return y

def theodorsen(q, a):
    """
    Função de Theodorsen para coeficientes aerodinâmicos

    Parameters:
    q : float - frequência reduzida
    a : float - parâmetro de posição do eixo elástico

    Returns:
    l_h, l_t, m_h, m_t : floats - coeficientes de Theodorsen
    """
    # Calcular a função de Theodorsen
    C_k = Ck(q)

    # Coeficientes aerodinâmicos baseados na teoria de Theodorsen
    # Estes são os coeficientes típicos para flutter analysis
    l_h = -np.pi * q * 2 + 2 * np.pi * C_k * q * 1j
    l_t = np.pi * q * 2 * (0.5 + a) + 2 * np.pi * C_k * (0.5 + a) * 1j
    m_h = np.pi * q * 2 * (0.5 + a) + 2 * np.pi * C_k * (0.5 + a) * 1j
    m_t = -np.pi * q * 2 * (1/8 + a**2) - 2 * np.pi * C_k * a * 1j

    return l_h, l_t, m_h, m_t

# Dados
b = 0.125
m = 1.542
l = 0.5
m_f = 2.548
I = 0.0072
k_h = 4.26
k_t = 5.08
c = -0.5
c_t = 0.088
c_h = 0.0035
x_t = 0.256

rho = 1.2754
k = np.linspace(1e-4, 1, 100)

# Adimensional
r = np.sqrt(I / (m * b ** 2))
mu_t = m / (rho * np.pi * b ** 2)
mu_h = (m + m_f) / (rho * np.pi * b**2)
omg_h = np.sqrt(k_h / (m + m_f))
omg_t = np.sqrt(k_t / m)
sigma = omg_h / omg_t
zeta_h = c_h / (2 * np.sqrt(k_h * m))
zeta_t = c_t / (2 * np.sqrt(k_t * m))

# Inicialização dos arrays de resultados
resultados = []
g_h = np.zeros(len(k))
g_t = np.zeros(len(k))
k_h_array = np.zeros(len(k))
k_t_array = np.zeros(len(k))
U = np.zeros(len(k))

# Definição da variável simbólica
lbd = sp.Symbol('(w_t/w)', complex=True)
import tqdm
# Loop principal
for idx, k_val in tqdm.tqdm(enumerate(k)):
    # Obter coeficientes de Theodorsen
    l_h, l_t, m_h, m_t = theodorsen(k_val, c)

    # Construir matriz M
    M11 = mu_h * (1 - sigma**2 * lbd**2 * mu_t/mu_h + 2j*zeta_h*mu_t/mu_h*sigma*lbd) + complex(l_h)
    M12 = mu_t * x_t + complex(l_t)
    M21 = mu_t * x_t + complex(m_h)
    M22 = mu_t * r**2 * (1 - lbd**2 + 2j*zeta_t*lbd) + complex(m_t)

    M = sp.Matrix([[M11, M12], [M21, M22]])

    # Calcular determinante
    D = M.det()

    # Resolver para encontrar as raízes
    # 2 raízes complexas (2 pares conjugados)
    # parte real é uma aproximação para w_t/w
    # parte imaginária é relacionada ao amortecimento do GDL
    solucoes = sp.solve(D, lbd)

    # Converter para números complexos
    solucoes_numericas = np.array([complex(sol.evalf()) for sol in solucoes]) 
    freq = omg_t / solucoes_numericas
    
    modo = np.argmin(np.abs(freq.imag))
    w_sel = np.abs(freq[modo].real) # conjugado positivo 

    # Armazenar resultados
    resultados.extend([sol for sol in solucoes_numericas if sol.real > 0])

    if len(solucoes_numericas) >= 2:
        g_h[idx] = solucoes_numericas[0].imag
        g_t[idx] = solucoes_numericas[1].imag
        k_h_array[idx] = solucoes_numericas[0].real
        k_t_array[idx] = solucoes_numericas[1].real
    U[idx] = w_sel * b / k_val

# Gráficos
# Gráfico 1: Amortecimentos
plt.figure(figsize=(10, 6))
plt.plot(U, g_t, 'b-', label='Pitch', linewidth=2)
plt.plot(U, g_h, 'r-', label='Plunge', linewidth=2)
plt.xlabel('Velocidade (U) [m/s]')
plt.ylabel('Amortecimento g [-]')
plt.grid(True)
plt.legend()
plt.title('Amortecimentos')
plt.tight_layout()
plt.savefig("amortecimentos.png")
plt.show()

# Gráfico 2: Frequência
plt.figure(figsize=(10, 6))
plt.plot(U, k_t_array, 'b-', label='Pitch', linewidth=2)
plt.plot(U, k_h_array, 'r-', label='Plunge', linewidth=2)
plt.xlabel('Velocidade (U) [m/s]')
plt.ylabel('Frequência reduzida (k) [-]')
plt.grid(True)
plt.legend()
plt.title('Frequência')
plt.tight_layout()
plt.savefig("frequencia.png")
plt.show()

print("Conversão concluída!")
print(f"Número de pontos processados: {len(k)}")
print(f"Faixa de velocidades: {U.min():.2f} - {U.max():.2f} m/s")