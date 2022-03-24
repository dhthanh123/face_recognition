import hashlib
string_dt = "123456789"
input_hash = string_dt.strip().encode("utf-8")
a= hashlib.md5(input_hash)
print(a.hexdigest())

def md5_encode(string):
    hash = string.strip().encode("utf-8")
    hash = hashlib.md5((hash))
    return hash.hexdigest()

if __name__=="__main__":
    d = md5_encode("123456789")
    print(d)

