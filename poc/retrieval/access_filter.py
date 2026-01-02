def filter_by_role(chunks, role):

    if role == "admin":
        return chunks  

    return [c for c in chunks if c["access"] == "public"]
