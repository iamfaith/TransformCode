try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math


class AddTryCatch(BaseOperator):
    def __init__(self, language: str):
        super(AddTryCatch, self).__init__(language)

    def modifyToken(self, token):
        tryStr = "try {\n" + token + "\n" + "} catch (Exception ex) {\n" + "ex.printStackTrace();\n" + "}"
        return tryStr.encode()
        # return token.encode()

    def addTryCatch(self, code_snippet, ratio=1.0):

        tree = self.parse(code_snippet)
        statements = self.get_nodes(code_snippet, tree.root_node)

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

        # sort index
        origin_pos.sort(key=lambda x: x[1], reverse=False)
        # print(origin_pos)
        #  reverse
        for i, start, end in reversed(origin_pos):
            tokenByte = self.getTokenByte(statements[i][0], oriContent)
            content = content[:start] + self.modifyToken(tokenByte.decode()) + content[end:]
            # print(content)
            # print(i, start, end, tokenByte.decode())
        try:
            content = content.decode('utf-8', 'ignore')
            return statements, content
        except Exception as e:
            # print("addTryCatch error", content)
            return statements, code_snippet

    # def getTokenByte(self, child, content):
    #     return content[child.start_byte:child.end_byte]

    # def get_statement_nodes(self, code, root):
    #     queue = [root]
    #     statements = []
    #     content = code.encode()
    #     statement_types = ['expression_statement']
    #     ignore_types = ['labeled_statement']
    #     while queue:
    #         current_node = queue.pop(0)
    #         for child in current_node.children:
    #             child_type = str(child.type)
    #             if child_type in ignore_types:
    #                 continue
    #             token = self.getTokenByte(child, content).decode()
    #             if child_type in statement_types:
    #                 # if self.checkChild(child):
    #                 # captures = self.query.captures(child)
    #                 statements.append((child, token))
    #             else:
    #                 # don't append child
    #                 queue.append(child)

    #             # captures = None
    #             # print(child.sexp(), token )
    #             # print(child, token)
    #     return statements


def main():
    addTryCatch = AddTryCatch(language="java")

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

    code = '''public static String var1(String var2) { MessageDigest var3 = null; try { var3 = MessageDigest.getInstance("MD5"); } catch (NoSuchAlgorithmException var7) { log.error(e.getMessage() + "Could not find the requested hash method: ");
System.out.println("Dummy"); } byte[] var4 = md5.digest(); public static String var1 int var6 = 0;while(var4.var8 > var6){ hexString.append(Integer.toHexString(var4[var6] & 0xFF));} return hexString.toString(); }'''

    sample_codes = [code]
    for sample_code in sample_codes:
        statements, code = addTryCatch(sample_code)
        print("-" * 50)
        for i in statements:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
