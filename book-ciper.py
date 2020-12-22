import random


from chargetter import chargetter



def addchar(a, b, negative_a=False, negative_b=False):
    a = ord(a) - ord('a')
    b = ord(b) - ord('a')
    if negative_a:
        a = -a
    if negative_b:
        b = -b
    return chr((a + b + 52) % 26 + ord('a'))

def computexorstream(stream_a, stream_b, negative_a=False, negative_b=False):
    for a, b in zip(stream_a, stream_b):
        yield addchar(a, b, negative_a, negative_b)





def bookcipher(bookchars = None, messagechars = None, cipherchars = None):
    if cipherchars == None:
        yield from computexorstream(bookchars, messagechars)
    if bookchars == None:
        yield from computexorstream(cipherchars, messagechars, negative_b=True)
    if messagechars == None:
        yield from computexorstream(cipherchars, bookchars, negative_b=True)
        



if __name__ == '__main__':

    bookchars = chargetter(filename='War-peace.txt')
    messagechars = chargetter(filename='input-str.txt')
    cipherchars = chargetter(filename='cipher-str.txt')

    # pop random number of characters off the front of the book
    # so as to start at a random index
    offset = random.randint(0, 0)
    for i in range(0, offset):
        next(bookchars, None)

    book = ''.join(bookcipher(messagechars=messagechars, cipherchars=cipherchars))

    print(book, offset)