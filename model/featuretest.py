import interpolation


def main():
  quint = interpolation.QuinticPolynomial(5, 0.2, 0, 0, 10, 5, 0, 0)
  t = 5.0
  while t <= 11.0:
    print(t, quint.get_position(t))
    t += 0.2


if __name__ == '__main__':
  main()
