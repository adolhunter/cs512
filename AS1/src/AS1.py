import numpy as np
from numpy import linalg as LA

def sectionA():
	print("\nAnswers of sectionA:\n")
	A, B, C = np.matrix('1; 2; 3'), np.matrix('4; 5; 6'), np.matrix('-1; 1; 3')
	#1
	print("1. 2A - B =\n", (2 * A - B))
	#2
	normA = LA.norm(A)
	angleA = np.rad2deg(np.arccos(1 / normA))
	print("2. |A| = %f ,the angle of A relative to the positive X axis is %f" % (normA,angleA))
	#3
	unitA = 1 / normA * A
	print("3. The unit vector in the direction of A is:\n", unitA)
	#4
	a1 = float(A[0])
	a2 = float(A[1])
	a3 = float(A[2])
	dirctionC = (a1, a2, a3) / normA
	print("4. The dircetion cosines of A are:", dirctionC)
	#5
	AdotB = float(np.dot(A.T, B))
	BdotA = float(np.dot(B.T, A))
	print("5. AdotB = ", AdotB,"BdotA = ", BdotA)
	#6
	normB = LA.norm(B)
	angleAB = np.rad2deg(np.arccos(AdotB / (normA * normB)))
	print("6. the angle between A and B:", angleAB)
	#7
	x = 1
	y = 1
	z = -(x + 2 * y) / 3
	print("7. A vector which is perpendicular to A is:", (x, y ,z))
	#8
	b1 = float(B[0])
	b2 = float(B[1])
	b3 = float(B[2])
	AxB = np.cross(A.T, B.T)
	BxA = - AxB
	print("8. AxB = ", AxB, "BxA = ", BxA)
	#9
	print("9. A vector perpendicular to A and B is: A x B = ", AxB)
	#10
	AB = np.column_stack([A, B])
	x = np.linalg.lstsq(AB, C)
	##len(x[0]) > 0 means 3A - B - C = 0 has answers.
	if len(x[0] > 0):
		print("10. A, B, C is linear dependency.")
	else:
		print("10. A, B, C is not linear dependency.")
	#11
	AtB = np.dot(A.T, B)
	BtA = np.dot(A, B.T)
	print("11. AtB = ", AtB, "\nBtA = ", BtA)


def sectionB():
	print("\nAnswers of sectionB:\n")
	A, B, C = np.matrix('1, 2, 3; 4, -2, 3; 0, 5, -1'), np.matrix('1, 2, 1; 2, 1, -4; 3, -2, 1'), np.matrix('1, 2, 3; 4, 5, 6; -1, 1, 3')
	#1
	print("1. 2A - B = ", 2 * A - B)
	#2
	AB = np.dot(A, B)
	BA = np.dot(B, A)
	print("2. AB = ", AB, "\nBA = ", BA)
	#3
	ABt =(AB).T
	BtAt = ABt
	print("3. (AB)t = ", ABt, "\nBtAt = ", BtAt)
	#4
	detA = np.linalg.det(A)
	print("4. |A| = ", detA, "|C| = 0, Because of the A10, we know the three row of this matrix is linear dependency.")
	#5
	print("5. The matrix in which the row vectors form an orthogonal set is B")
	for i in range(0, len(B)):
		for j in range(i + 1, len(B)):
			if np.dot(B[i], B[j].T) == 0:
				print("orthogonal set: ", B[i],B[j].T)

	#6
	invA = np.linalg.inv(A)
	invB = np.linalg.inv(B)
	print("6. invA =", invA,"\n invB =", invB)


def sectionC():
	print("\nAnswers of sectionC:\n")
	A, B = np.matrix('1, 2; 3, 2'), np.matrix('2, -2; -2, 5')
	#1
	eigvalA, V = np.linalg.eig(A)
	print("1. eigenvalue of A = ",eigvalA, "\n eigenvector of A = ", V)
	#2
	invVAV = np.round(np.dot(np.dot(np.linalg.inv(V), A), V))
	print("2. invVAV = ", invVAV)
	#3
	v1 = np.matrix([V[0, 0],V[1, 0]])
	v2 = np.matrix([V[0, 1],V[1, 1]])
	dotEigA = np.dot(v1, v2.T)
	print("3. dot product between the eigenvector of A is ", dotEigA)

	#4
	eigvalB, eigB = np.linalg.eig(B)
	v3 = np.matrix([eigB[0, 0],eigB[1, 0]])
	v4 = np.matrix([eigB[0, 1],eigB[1, 1]])
	dotEigB = np.dot(v3, v4.T)
	print("4. dot product between the eigenvector of B is ", dotEigB)
	#5
	print("5. The eignvectors of B are orthogonal matrix, because B is a symmetric real matrix.")

def main():
	sectionA()
	sectionB()
	sectionC()


if __name__ == '__main__':
	main()