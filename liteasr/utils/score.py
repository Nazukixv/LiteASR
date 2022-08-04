"""Score."""


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""

    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    curr = list(range(n + 1))
    for i in range(1, m + 1):
        prev, curr = curr, [i] + [0] * n
        for j in range(1, n + 1):
            insert, delete = prev[j] + 1, curr[j - 1] + 1
            change = prev[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            curr[j] = min(insert, delete, change)

    return curr[n]
