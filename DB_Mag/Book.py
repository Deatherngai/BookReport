import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Utility function to compute similarity
def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

# Use a service account.
def QueryBook(book):
    cred = credentials.Certificate('DB_Mag/SDK2.json')
    if not firebase_admin._apps:
        app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    docs = (db.collection("LMS").document("Tables").collection("Book").stream())
    values = []
    for doc in docs:
        rate = similar(book,doc.to_dict()['BookName_EN'])
        if(rate>0.5):
            row = {"sim_rate":round(rate, ndigits=2), "data":doc.to_dict()}
            values.append(row)

    return values