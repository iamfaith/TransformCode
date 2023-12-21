import tree_sitter
from tree_sitter import Node
from typing import List
try:
    from base_operator import BaseOperator
    from ast_parser import ASTParser
except:
    from .base_operator import BaseOperator
    from .ast_parser import ASTParser
class CommentDeletion(BaseOperator):

    def __init__(self, language: str):
        super(CommentDeletion, self).__init__(language)

    def match_from_span(self, node: Node, lines: List) -> str:
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        char_start = node.start_point[1]
        char_end = node.end_point[1]
        if line_start != line_end:
            return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
        else:
            return lines[line_start][char_start:char_end]

    def delete_from_span(self, nodes, lines: List) -> str:
        new_lines = []
        for i in range(len(lines)):
            line_nodes = []
            for node in nodes:
                line_start = node.start_point[0]
                line_end = node.end_point[0]
                char_start = node.start_point[1]
                char_end = node.end_point[1]
                if i in range(line_start,line_end+1):
                    if i == line_start and i == line_end:
                        line_nodes.append([char_start, char_end])
                    elif line_start == i:
                        line_nodes.append([char_start, len(lines[i])])
                    elif line_end == i:
                        line_nodes.append([0, char_end])
                    else:
                        line_nodes.append([0, len(lines[i])])

            new_line = ""
            for j in range(len(lines[i])):
                inside = False
                for s in line_nodes:
                    # print(s)
                    if j in range(s[0], s[1]):
                        inside = True
                        break
                if not inside:
                    new_line = new_line + lines[i][j]
            new_lines.append(new_line)
        return '\n'.join(new_lines)
        # return ''.join(new_lines)

    def get_comment_nodes(self, tree_splitted_code, root):
        queue = [root]
        comments = []
        comment_types = ['line_comment', 'block_comment', 'comment']
        while queue:
          current_node = queue.pop(0)
          for child in current_node.children:
             child_type = str(child.type)
             if child_type in comment_types:
                 comments.append(child)
             queue.append(child)
        return comments


    def commentDeletion(self, code_snippet):

        tree = self.parse(code_snippet)
        tree_splitted_code = code_snippet.split("\n")

        cmts = self.get_comment_nodes(tree_splitted_code, tree.root_node)
        code = self.delete_from_span(cmts, tree_splitted_code)

        return cmts, code


def main():
    
    comment_deletion_operator = CommentDeletion(language="rust")

    code_with_comment = """
    // this is a comment
    fn main() {
        int i = 1;
    /* this is another
       comment
       */
       i = 20;
    }
    """

    code_without_comment = comment_deletion_operator(code_with_comment)
    print(code_without_comment)

if __name__ == "__main__":
    main()
