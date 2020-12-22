import random

from chargetter import chargetter










def generate_text(messagefile, keyfile, num_samples, length, randomoffset=100):

    text = []
    for i in range(0, num_samples):
        messagechars = chargetter(filename=messagefile)
        keychars = chargetter(filename=keyfile)

        # pop random number of characters off the front of the book
        # so as to start at a random index
        offset = random.randint(0, randomoffset)
        for i in range(0, offset):
            next(messagechars, None)

        # pop random number of characters off the front of the book
        # so as to start at a random index
        offset = random.randint(0, randomoffset)
        for i in range(0, offset):
            next(keychars, None)
        
        message = ''
        key = ''
        for i in range(0, length):
            message += next(messagechars, None)
            key += next(keychars, None)
        
        text.append((message, key))

    return text



