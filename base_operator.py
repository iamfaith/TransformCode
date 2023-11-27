try:
    from ast_parser import ASTParser
except:
    from .ast_parser import ASTParser

class BaseOperator():

    def __init__(self, language: str):

        self.parser = ASTParser(language=language)
        self.lang = self.parser.set_language(language)

    def parse(self, code_snippet):
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        return tree

    def getTokenByte(self, child, content):
        return content[child.start_byte:child.end_byte]

    def getTokenContent(self, node):
        content = ""
        for n in node.children:
            content += n.text.decode()
        return content

    def get_nodes(
            self, code, root, debug=False, capture_types=['expression_statement'],
            ignore_types=['labeled_statement'], cb=None, focus=None, loopChildren=False):
        queue = [root]
        statements = []
        content = code.encode()
        idx = 0
        while queue:
            current_node = queue.pop(0)
            for child in current_node.children:
                child_type = str(child.type)
                if ignore_types is not None and child_type in ignore_types:
                    continue
                token = self.getTokenByte(child, content).decode()
                if capture_types is not None and child_type in capture_types:
                    # if self.checkChild(child):
                    # captures = self.query.captures(child)
                    statements.append((child, token))
                    if loopChildren:
                        queue.append(child)
                else:
                    # don't append child
                    if ignore_types is None or capture_types is None or (focus is not None and child_type in focus):
                        statements.append((child, token))
                    if cb is None or (cb is not None and cb(child, token) == True):
                        queue.append(child)

                # captures = None
                # print(child.sexp(), token )
                if debug:
                    print(idx, child, token)
                    idx += 1
        return statements

    def __call__(self, *args, **kvargs):
        # self.swapCondition(code)
        func_name = self.__class__.__name__
        func_name = func_name[0].lower() + func_name[1:]

        suffix = ''
        if 'suffix' in kvargs:
            suffix = kvargs['suffix']
        
        # print(func_name)
        if hasattr(self, func_name + suffix):
            func = getattr(self, func_name + suffix)
        else:
            func = getattr(self, func_name)
        if 'suffix' in kvargs:
            del kvargs['suffix']
        return func(*args, **kvargs)

    def getNodeType(self, node):
        return str(node.type)

    def getChildType(self, node, idx=0):
        return self.getNodeType(node.children[idx])
