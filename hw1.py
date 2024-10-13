import math

def parse_equations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    A = []
    B = []

    for line in lines:
        line = line.replace('= ', '').replace('=', '')
        equation_parts = line.split()
        B.append(int(equation_parts[-1]))

        coefficients = []
        for term in equation_parts[:-1]:
            if 'x' in term:
                coef = term.replace('x', '')
                coefficients.append(int(coef) if coef not in ['', '+', '-'] else (1 if coef != '-' else -1))
            elif 'y' in term:
                coef = term.replace('y', '')
                coefficients.append(int(coef) if coef not in ['', '+', '-'] else (1 if coef != '-' else -1))
            elif 'z' in term:
                coef = term.replace('z', '')
                coefficients.append(int(coef) if coef not in ['', '+', '-'] else (1 if coef != '-' else -1))

        A.append(coefficients)

    return A, B

def trace_matrix(A):
    trace = A[0][0] + A[1][1] + A[2][2]
    return trace


def norm_vector(B):
    norm = math.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    return norm


def transpose_matrix(A):
    transpose = [[A[j][i] for j in range(3)] for i in range(3)]
    return transpose


def matrix_vector_multiply(A, B):
    result = [
        A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2],
        A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2],
        A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2]
    ]
    return result


def generate_modified_matrix(A, B, column_index):
    modified_matrix = [row[:] for row in A]
    for i in range(3):
        modified_matrix[i][column_index] = B[i]
    return modified_matrix


def cramer_solution(A, B):
    det_A = determinant_3x3(A)
    if det_A == 0:
        raise ValueError("Sistemul nu are soluție unică, determinanții sunt 0.")

    Ax = generate_modified_matrix(A, B, 0)
    Ay = generate_modified_matrix(A, B, 1)
    Az = generate_modified_matrix(A, B, 2)

    det_Ax = determinant_3x3(Ax)
    det_Ay = determinant_3x3(Ay)
    det_Az = determinant_3x3(Az)

    x = det_Ax / det_A
    y = det_Ay / det_A
    z = det_Az / det_A

    return x, y, z


def minor(A, row, col):
    return [ [A[i][j] for j in range(len(A)) if j != col] for i in range(len(A)) if i != row ]
def determinant_2x2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

def determinant_3x3(A):
    if len(A) != 3 or len(A[0]) != 3 or len(A[1]) != 3 or len(A[2]) != 3:
        raise ValueError("Matricea trebuie să fie de dimensiune 3x3.")
    det = (
            A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
            A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
            A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )
    return det


def cofactor(A):
    cofactors = []
    for i in range(len(A)):
        row = []
        for j in range(len(A)):
            m = minor(A, i, j)
            c = ((-1) ** (i + j)) * (determinant_2x2(m) if len(m) == 2 else determinant_3x3(m))
            row.append(c)
        cofactors.append(row)
    return cofactors


def adjugate(A):
    return transpose_matrix(cofactor(A))

def inverse(A):
    det_A = determinant_3x3(A)
    if det_A == 0:
        raise ValueError("Matricea nu are inversă, determinanții sunt 0.")
    adj_A = adjugate(A)
    inv_A = [[adj_A[i][j] / det_A for j in range(len(adj_A))] for i in range(len(adj_A))]
    return inv_A

if __name__ == "__main__":
    A, B = parse_equations('ex.txt')

    print("Matricea A:", A)
    print("Vectorul B:", B)

    print("Determinantul lui A:", determinant_3x3(A))
    print("Trace-ul lui A:", trace_matrix(A))
    print("Norma vectorului B:", norm_vector(B))
    print("Transpusa lui A:", transpose_matrix(A))
    print("A * B:", matrix_vector_multiply(A, B))

    # Soluția folosind regula lui Cramer
    try:
        x, y, z = cramer_solution(A, B)
        print(f"Soluția sistemului prin regula lui Cramer: x = {x}, y = {y}, z = {z}")
    except ValueError as e:
        print(e)

    # Soluția folosind inversa matricei
    try:
        inv_A = inverse(A)
        X = matrix_vector_multiply(inv_A, B)
        print(f"Soluția sistemului prin inversa matricei: x = {X[0]}, y = {X[1]}, z = {X[2]}")
    except ValueError as e:
        print(e)