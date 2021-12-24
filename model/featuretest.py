import interpolation


def main():
  quint = interpolation.QuinticPolynomial(0, -1.5164, 0, 0, 45.625, -3.7626, 0, 0)
  t = 0.0
  while t <= 46.0:
    print(t, quint.get_position(t))
    t += 0.125


if __name__ == '__main__':
  main()
