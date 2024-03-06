
import difflib


string1 = "Leading with Principle"
string1 = string1.lower()
string2 = "leading with principle"
string3 = "Leading with Adversity"
string3 = string3.lower()
string4 = "leading from adversity"
string5 = "Let's Talk About Body Boundaries, Consent and Respect"
string5 = string5.lower()
string6 = "letstalk about body boundaries consent respect"
string7 = "Diary of a Wimpy Kid #02 Rodrick Rules"
string7 = string7.lower()
string8 = "wimpylidd rick rodr rules te"

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

rate1 = similar(string1,string2)
rate2 = similar(string3,string4)
rate3= similar(string5,string6)
print(string5)
rate4 = similar(string7,string8)

#print(rate1)
#print(rate2)
#print(rate3)
#print(rate4)

from simhash import Simhash


def simhash_similarity(text1, text2):
    """
    :param text1: 文本1
    :param text2: 文本2
    :return: 返回两篇文章的相似度
    """
    aa_simhash = Simhash(text1)
    bb_simhash = Simhash(text2)
    max_hashbit = max(len(bin(aa_simhash.value)), (len(bin(bb_simhash.value))))
    # 汉明距离
    distince = aa_simhash.distance(bb_simhash)
    similar = 1 - distince / max_hashbit
    return similar


if __name__ == '__main__':
    print(simhash_similarity('在历史上有著许多数学发现', '在历史上有著许多科学发现'))