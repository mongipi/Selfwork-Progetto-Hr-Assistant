def leggi_prime_righe(file_path, n=20):
    righe = []

    with open(file_path, "r", encoding="utf-8") as file:
        for i, riga in enumerate(file):
            if i < n:
                righe.append(riga.strip())
            else:
                break

    return "\n".join(righe)