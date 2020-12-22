

def chargetter(filename = None, allowed_characters = None, skip=0):
    if allowed_characters == None:
        allowed_characters = 'abcdefghijklmnopqrstuvwxyz'
    allowed_characters = set(allowed_characters.lower() + allowed_characters.upper())
    with open(filename, 'r') as f:
        f.seek(skip)
        while True:
            next_char = f.read(1)
            if next_char == '':
                break
            if next_char not in allowed_characters:
                continue
            yield next_char.lower()
