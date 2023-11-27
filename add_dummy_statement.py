try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math
import string


class AddDummyStatement(BaseOperator):

    def __init__(self, language: str):
        super(AddDummyStatement, self).__init__(language)

        N = 6
        self.funcName = ''.join(random.choice(string.ascii_lowercase
                                              ) + random.choice(string.ascii_lowercase + string.digits)
                                for _ in range(N - 1))
        self.className = ''.join(random.choice(string.ascii_uppercase
                                               ) + random.choice(string.ascii_lowercase + string.digits)
                                 for _ in range(N - 1)) 
        self.dummyStatement = [
            "System.out.println(\"Dummy\");", "if (1 > 2) { System.out.println(\"Dummy\"); }",
            "if (2 > 1) { System.out.println(\"Dummy\"); }",
            "if (1 > 2)  { System.out.println(\"Dummy1\"); } else  { System.out.println(\"Dummy2\"); }",
            "if (false) { int temp = 100; }", self.className + "." + self.funcName + "();"]
        self.dummyClass = '''class ''' + self.className + ''' {
    public static void ''' + self.funcName + '''() {
    }
}
'''
        if "c" == language or "cpp" == language:
            self.dummyClass = '''public void ''' + self.funcName + '''() {
        }
    '''     
            self.className = ''
            self.dummyStatement.pop()
            self.dummyStatement.append(self.funcName + "();")
        self.addDummyClass = False

    def _addDummy(self, token):
        dummyIdx = random.randint(0, len(self.dummyStatement) - 1)
        dummyStmt = self.dummyStatement[dummyIdx]
        if dummyIdx == len(self.dummyStatement) - 1:
            self.addDummyClass = True
        return (token + '\n' + dummyStmt).encode()

    def addDummyStatement(self, code_snippet, ratio=1.0):

        tree = self.parse(code_snippet)
        statements = self.get_nodes(code_snippet, tree.root_node, debug=False)

        oriContent = code_snippet.encode()
        content = code_snippet.encode()
        total = len(statements) - 1
        sample_num = math.ceil((total + 1) * ratio)
        ran = [random.randint(0, total) for i in range(sample_num)]
        ran = set(ran)
        origin_pos = []
        for idx in ran:
            child = statements[idx][0]
            origin_pos.append((idx, child.start_byte, child.end_byte))

        total = len(statements) - 1
        self.addDummyClass = False
        origin_pos.sort(key=lambda x: x[1], reverse=False)
        #  reverse
        for i, start, end in reversed(origin_pos):
            tokenByte = self.getTokenByte(statements[i][0], oriContent)
            content = content[:start] + self._addDummy(tokenByte.decode()) + content[end:]
            # print(i, start, end, tokenByte.decode())
        if self.addDummyClass:
            # add function define
            content = self.dummyClass.encode() + content
        try:
            return statements, content.decode('utf-8', 'ignore')
        except Exception as e:
            return statements, code_snippet


def main():
    addDummy = AddDummyStatement(language="java")

    sample_codes = [
        """
        public class Main {
          public static void main(String[] args) {
            int a = 15;
            Test.func1(a);
            circle.circumference(100, 20);
            if (b > 12) {
                label123: println("continuing");
                System.out.println("hello");
                continue;
                break;
                ;
                return 1;
            }
          }
        }
        """,
        """
        void bubbleSort(int arr[], int n)
        {
           int i, j;
           for (i = 0; i < n-1; i++)     
         
               // Last i elements are already in place  
               for (j = 0; j < n-i-1; j++)
                   if (arr[j] > arr[j+1])
                      swap(&arr[j], &arr[j+1]);
        }
        """
    ]
    source = 'public static String var1(String var2) { MessageDigest var3 = null; try { var3 = MessageDigest.getInstance("MD5"); md5.reset(); md5.update(value.getBytes()); } catch (NoSuchAlgorithmException var7) { log.error("Could not find the requested hash method: " + e.getMessage()); } byte[] var4 = md5.digest(); StringBuffer var5 = new StringBuffer(); for (int var6 = 0; var6 < var4.var8; var6++) { hexString.append(Integer.toHexString(0xFF & var4[var6])); } return hexString.toString(); }'
    # sample_codes = [source]
    for sample_code in sample_codes:
        statements, code = addDummy(sample_code)
        print("-" * 50)
        for i in statements:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
