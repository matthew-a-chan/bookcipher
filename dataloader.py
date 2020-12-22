import random

from chargetter import chargetter










def generate_text(messagefile, keyfile, num_samples, length, randomoffset=100):

    for i in range(0, num_samples):
        messagechars = chargetter(filename=messagefile, skip=random.randint(0, randomoffset))
        keychars = chargetter(filename=keyfile, skip=random.randint(0, randomoffset))
        
        message = ''
        key = ''
        for i in range(0, length):
            message += next(messagechars, None)
            key += next(keychars, None)
        
        yield(message, key)



