from sympy import symbols, Matrix, zeros

# Определяем переменные
a = symbols('a11 a12 a13 a14 a21 a22 a31 a33 a35 a41 a44 a45 a53 a54 a55 a56 a57 a58 a65 a66 a68 a75 a77 a78 a85 a86 a87 a88')

# Создаём нулевую матрицу 8x8
A = zeros(8, 8)

# Заполняем матрицу
A[0, :] = Matrix([a[0], a[1], a[2], a[3], 0, 0, 0, 0])
A[1, :] = Matrix([a[4], a[5], 0, 0, 0, 0, 0, 0])
A[2, :] = Matrix([a[6], 0, a[7], 0, a[8], 0, 0, 0])
A[3, :] = Matrix([a[9], 0, 0, a[10], a[11], 0, 0, 0])
A[4, :] = Matrix([0, 0, a[12], a[13], a[14], a[15], a[16], a[17]])
A[5, :] = Matrix([0, 0, 0, 0, a[18], a[19], 0, a[20]])
A[6, :] = Matrix([0, 0, 0, 0, a[21], 0, a[22], a[23]])
A[7, :] = Matrix([0, 0, 0, 0, a[24], a[25], a[26], a[27]])

# Создаём столбец правых частей
b = zeros(8, 1)  # Нулевая матрица (правые части уравнений)

# Решаем систему
x = A.solve(b)
print("Решение системы:")
print(x)
