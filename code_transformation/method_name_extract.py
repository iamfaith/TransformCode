try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math
import re

class MethodNameExtract(BaseOperator):

    def __init__(self, language: str):
        super(MethodNameExtract, self).__init__(language)
        self.methodNames = ["method_declaration", 'constructor_declaration']
        self.declaration = ['local_variable_declaration']  # java
        if "c" == language or "cpp" == language:
            self.methodNames = ["function_definition"]
            self.declaration = ["declaration"]
        self.language = language

    def methodNameExtract(self, code_snippet):
        tree = self.parse(code_snippet)
        methods = self.get_nodes(code_snippet, tree.root_node, debug=False, ignore_types=[
                                 "comment"], capture_types=self.methodNames)
        oriContent = code_snippet.encode()
        content = code_snippet.encode()

        ret = []
        for method_idx, (method, token) in enumerate(reversed(methods)):
            method_name = ''
            if self.language == 'c' or self.language == 'cpp':
                _methods = self.get_nodes(code_snippet, method, debug=False, capture_types=['function_declarator'])
                if len(_methods) > 0:
                    method, _ = _methods[0]
                else:
                    continue
            for node in method.children:
                if self.getNodeType(node) == "identifier":
                    method_name = node.text.decode()
                    break
            method_name = splitToSubtokens(method_name)
            ret.append((method_name, token))
        return ret


def normalizeName(original, defaultString):
    original = original.lower().replace("\\n", "").replace(
        "//s+", "").replace("[\"',]", "").replace("\\P{Print}", "")
    stripped = re.sub("[^A-Za-z]", "", original)
    if len(stripped) == 0:
        carefulStripped = original.replace(" ", "_")
        if len(carefulStripped) == 0:
            return defaultString
        else:
            return carefulStripped
    else:
        return stripped


def splitToSubtokens(str1):
    str2 = str1.strip()
    return [normalizeName(s, "") for s in re.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+", str2) if len(s) > 0]

    # public static String normalizeName(String original, String defaultString) {
    # 	original = original.toLowerCase().replaceAll("\\\\n", "") // escaped new
    # 																// lines
    # 			.replaceAll("//s+", "") // whitespaces
    # 			.replaceAll("[\"',]", "") // quotes, apostrophies, commas
    # 			.replaceAll("\\P{Print}", ""); // unicode weird characters
    # 	String stripped = original.replaceAll("[^A-Za-z]", "");
    # 	if (stripped.length() == 0) {
    # 		String carefulStripped = original.replaceAll(" ", "_");
    # 		if (carefulStripped.length() == 0) {
    # 			return defaultString;
    # 		} else {
    # 			return carefulStripped;
    # 		}
    # 	} else {
    # 		return stripped;
    # 	}
    # }

    # public static ArrayList<String> splitToSubtokens(String str1) {
    # 	String str2 = str1.trim();
    # 	return Stream.of(str2.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
    # 			.filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
    # 			.filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
    # }


def main():
    filename = "/mnt/dataset/infercode/java-small/val-debug/Wav.java"
    with open(filename, "r") as f:
        code = f.read()
        # print(code)
        extract = MethodNameExtract(language='java')
        ret = extract(code)
        print(ret)


if __name__ == "__main__":
    main()

    # method_names = "skipFully"
    # # method_names = "seekToChunk"
    # ret = splitToSubtokens(method_names)
    # print(ret)
