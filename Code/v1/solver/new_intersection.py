def two_segments_intersection(s, t):
    """
    Find intersection between two segments s (endpoints a,b) and t (endpoints c,d)
    - s: (aX, aY, bX, bY)
    - t: (cX, cY, dX, dY)
    """
    aX, aY, bX, bY = s[0], s[1], s[2], s[3]
    cX, cY, dX, dY = t[0], t[1], t[2], t[3]

    A = (bX - aX) * (cY - dY) - (cX - dX) * (bY - aY)
    if not isclose(A, 0.0):
        As = (cX - aX) * (cY - dY) - (cX - dX) * (cY - aY)
        At = (bX - aX) * (cY - aY) - (cX - aX) * (bY - aY)
        ks = As / A
        kt = At / A
    else: 
        return False

    if 0.0-abs_tol <= ks <= 1.0+abs_tol and 0.0-abs_tol <= kt <= 1.0+abs_tol:
        intersection_X = (1 - ks) * aX + ks * bX
        intersection_Y = (1 - ks) * aY + ks * bY
    else: 
        return False

    return intersection_X, intersection_Y   