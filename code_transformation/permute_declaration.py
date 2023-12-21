try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math


class PermuteDeclaration(BaseOperator):

    def __init__(self, language: str):
        super(PermuteDeclaration, self).__init__(language)
        self.permuteClass = ["method_declaration"]
        self.declaration = ['local_variable_declaration'] # java
        if "c" == language or "cpp" == language:
            self.permuteClass = ["function_definition"]
            self.declaration = ["declaration"]

    def permuteDeclarationClass(self, code_snippet, ratio=1.0):

        tree = self.parse(code_snippet)
        methods = self.get_nodes(code_snippet, tree.root_node, debug=False, ignore_types=["comment"], capture_types=self.permuteClass)
        oriContent = code_snippet.encode()
        content = code_snippet.encode()
        for method_idx, (method, token) in enumerate(reversed(methods)):
            statements = self.get_nodes(code_snippet, method, debug=False,
                                        capture_types=self.declaration)
            if len(statements) < 2:
                continue
            origin_pos = []
            for child, token in statements:
                # for hack: (<Node kind=local_variable_declaration, start_point=(0, 0), end_point=(0, 25)>, 'public static String var1')
                if child.start_byte == 0:
                    continue
                origin_pos.append((child.start_byte, child.end_byte))

            random.shuffle(statements)

            total = len(statements) - 1
            origin_pos.sort(key=lambda x: x[1], reverse=False)
            for i, (start, end) in enumerate(reversed(origin_pos)):
                tokenByte = self.getTokenByte(statements[total - i][0], oriContent)
                content = content[:start] + tokenByte + content[end:]
        try:
            return None, content.decode('utf-8', 'ignore')
        except Exception as e:
            print(e)
            return None, code_snippet



    def permuteDeclaration(self, code_snippet, ratio=1.0):

        tree = self.parse(code_snippet)
        # type=local_variable_declaration
        statements = self.get_nodes(code_snippet, tree.root_node, debug=False,
                                    capture_types=['local_variable_declaration'])
        origin_pos = []
        for child, token in statements:
            # for hack: (<Node kind=local_variable_declaration, start_point=(0, 0), end_point=(0, 25)>, 'public static String var1')
            if child.start_byte == 0:
                continue
            origin_pos.append((child.start_byte, child.end_byte))

        # total = len(statements) - 1
        # sample_num = math.ceil((total + 1) * ratio)
        # ran = [random.randint(0, total) for i in range(sample_num)]
        # ran = set(ran)
        random.shuffle(statements)

        oriContent = code_snippet.encode()
        content = code_snippet.encode()
        total = len(statements) - 1
        origin_pos.sort(key=lambda x: x[1], reverse=False)
        for i, (start, end) in enumerate(reversed(origin_pos)):
            tokenByte = self.getTokenByte(statements[total - i][0], oriContent)
            content = content[:start] + tokenByte + content[end:]
            # print(i, start, end, tokenByte.decode())
            # print(content.decode('utf-8', 'ignore'))
        try:
            return statements, content.decode('utf-8', 'ignore')
        except Exception as e:
            return statements, code_snippet


def main():
    permuteDeclaration = PermuteDeclaration(language="java")

    sample_codes = [
        """
        public class Main {
          public static void main(String[] args) {
            int a = 15;
            int j = 20;
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
    sample_codes = [source]

    # (<Node kind=local_variable_declaration, start_point=(0, 0), end_point=(0, 25)>, 'public static String var1')
    #  can not permute
    for sample_code in sample_codes:
        statements, code = permuteDeclaration(sample_code)
        print("-" * 50)
        for i in statements:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
    # b = bytes("asd哈哈", 'utf-8')
    # s = b.decode()
    # s = s + '中国问'
    # print(s.encode())
    # print(b)
