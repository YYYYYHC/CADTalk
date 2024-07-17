'''Testing simple parsing examples'''
import re
import os
import unittest
from sca2d import Analyser
import pdb
from lark import Transformer, Token
from graphviz import Digraph
Primitive_Calls = ['sphere', 'cube', 'difference', 'square', 'cylinder', 'hull', 'linear_extrude', 'rotate_extrude', 'resize']
lines_to_label = []
nodes_to_label = []
variable_assignmented_lines = []

class treeNode:
    def __init__(self):
        self.data = None
        self.parent = None
        self.children = []
        self.labels = []

    def get_node(self, data, parent=None, children=None):
        self.data = data
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children
        return self

    def add_children(self, node):
        self.children.append(node)

def convert_to_tree(astNode, parentNode=None):
    nodeClass = type(astNode).__name__
    if nodeClass == 'dict':
        if 'line' in astNode:
            curNode = treeNode().get_node({'type': astNode['type'], 'line': astNode['line']})
        else:
            curNode = treeNode().get_node({'type': astNode['type']})
    elif nodeClass == 'Token':
        curNode = treeNode().get_node(astNode)

    curNode.parent = parentNode

    if parentNode is None:
        parentNode = curNode

    if 'children' in astNode:
        for astNodeChildren in astNode['children']:
            curNode.add_children(convert_to_tree(astNodeChildren, curNode))

    return curNode

def extract_labels(input_list):
    pattern = re.compile(r'//gt label:\s*(\S+)')
    return [match.group(1) for line in input_list for match in [pattern.search(line)] if match]

def label_lines(program_path, lines_to_label, placeholder, save_path):
    reverserOrder_lines_idxs = sorted(set(lines_to_label), reverse=True)

    with open(program_path) as f:
        program_lt = f.readlines()
        for idx in reverserOrder_lines_idxs:
            program_lt.insert(idx-1, f'{placeholder}\n')

    with open(save_path, 'w') as f:
        f.writelines(program_lt)

def visualize_ast_use_tree(node, graph=None, parent=None, name=None):
    if graph is None:
        graph = Digraph('AST', node_attr={'shape': 'box', 'height': '.1'})
        graph.attr(size='6,6')

    if name is None:
        name = 'root'

    nodeClass = type(node.data).__name__
    if nodeClass == 'dict':
        if 'line' in node.data:
            graph.node(name, label=node.data['type'] + ' line:' + str(node.data['line']) + f' labels: {node.labels}')
        else:
            graph.node(name, label=node.data['type'] + f' labels: {node.labels}')
    elif nodeClass == 'Token':
        graph.node(name, label=node.data + f' labels: {node.labels}')

    if parent is not None:
        graph.edge(parent, name)

    if nodeClass != 'Token' and 'type' in node.data.keys():
        if 'module_call' in node.data['type']:
            call_name = node.children[0].children[0].children[0].data
            if call_name in Primitive_Calls:
                lines_to_label.append(node.data['line'])
                nodes_to_label.append(node)
        if 'variable_assignment' in node.data['type']:
            variable_assignmented_lines.append(node.data['line'])

    if node.children is not None:
        for index, child in enumerate(node.children):
            child_name = f"{name}_{index}"
            visualize_ast_use_tree(child, graph, name, child_name)

    return graph

def ast_visualizer(node, graph=None, parent=None, name=None):
    if graph is None:
        graph = Digraph('AST', node_attr={'shape': 'box', 'height': '.1'})
        graph.attr(size='6,6')

    if name is None:
        name = 'root'

    nodeClass = type(node).__name__
    if nodeClass == 'dict':
        if 'line' in node:
            graph.node(name, label=node['type'] + ' line:' + str(node['line']))
        else:
            graph.node(name, label=node['type'])
    elif nodeClass == 'Token':
        graph.node(name, label=node)

    if parent is not None:
        graph.edge(parent, name)

    if nodeClass != 'Token' and 'type' in node.keys():
        if 'module_call' in node['type']:
            call_name = node['children'][0]['children'][0]['children'][0].value
            if call_name in Primitive_Calls:
                lines_to_label.append(node['line'])
                nodes_to_label.append(node)

    if 'children' in node:
        for index, child in enumerate(node['children']):
            child_name = f"{name}_{index}"
            ast_visualizer(child, graph, name, child_name)

    return graph

def find_matching_brackets(text):
    lines = text.split('\n')
    stack = []
    matches = []

    for line_number, line in enumerate(lines, start=1):
        if '{' in line:
            stack.append(line_number)
        elif '}' in line and stack:
            opening_line = stack.pop()
            matches.append((opening_line, line_number))

    return matches

def propagate_labels(nodes):
    for node in nodes:
        current = node
        while current.parent:
            current.parent.labels = list(set(current.parent.labels) | set(current.labels))
            current = current.parent

    root = nodes[0]
    while root.parent:
        root = root.parent

    return root

class SimpleASTTransformer_parsing(Transformer):
    def __default__(self, data, children, meta):
        if children:
            return {'type': data, 'children': children, 'line': meta.line}
        else:
            return {'type': data}

    def ITEM(self, item: Token):
        return {"type": item.type, "line": item.line}

class SimpleASTTransformer(Transformer):
    def __init__(self):
        self.function_definitions = {}

    def expand_function_definition(self, function_name):
        module_def = self.function_definitions.get(function_name)
        if module_def is not None:
            expanded_children = []
            for child in module_def:
                if child['type'] == 'module_call':
                    nested_function_name = child['children'][0]['children'][0]['children'][0].value
                    expanded_child = self.expand_function_definition(nested_function_name)
                    expanded_children.append(expanded_child)
                else:
                    expanded_children.append(child)
            return {'type': 'def_subtree', 'children': expanded_children, 'line': module_def[0]['line']}
        else:
            return

    def module_def(self, children):
        function_name = children[0]['children'][0]['children'][0].value
        self.function_definitions[function_name] = children
        return {'type': 'module_def', 'children': children, 'line': children[0]['line']}

    def module_call(self, children):
        function_name = children[0]['children'][0]['children'][0].value
        module_def = self.function_definitions.get(function_name)
        expanded_definition = self.expand_function_definition(function_name)

        if module_def is not None:
            return {'type': 'call_subtree', 'children': children + [{'type': 'def_subtree', 'children': [expanded_definition]}], 'line': children[0]['line']}
        else:
            return {'type': f'local_module_call:{function_name}', 'children': children, 'line': children[0]['line']}

    def __default__(self, data, children, meta):
        if children:
            return {'type': data, 'children': children, 'line': meta.line}
        else:
            return {'type': data}

    def ITEM(self, item: Token):
        return {"type": item.type, "value": item.value, "line": item.line}

ast_transformer = SimpleASTTransformer()

line_and_labels = []
func_labels_lines = {}
func_call_lines = []
func_call_nodes = []

def travel_and_get_passed_labels(passed_ast_tree):
    if type(passed_ast_tree.data).__name__ == 'Token':
        return

    if 'module_def' in passed_ast_tree.data['type']:
        module_name = passed_ast_tree.children[0].children[0].children[0].data.value
        func_labels_lines[module_name] = passed_ast_tree.data['line']

    if 'module_call' in passed_ast_tree.data['type']:
        module_name = passed_ast_tree.children[0].children[0].children[0].data.value
        func_call_nodes.append(passed_ast_tree)
        func_call_lines.append((module_name, passed_ast_tree.data['line']))

    if 'start' not in passed_ast_tree.data['type'] and 'line' in passed_ast_tree.data.keys():
        line_and_labels.append({'line': passed_ast_tree.data['line'], 'labels': passed_ast_tree.labels})

    for child in passed_ast_tree.children:
        travel_and_get_passed_labels(child)

    return

class FileAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.analyser = Analyser()
        self.scadfile_path = os.path.join('tests', 'scadfiles')

    def CADTalk_parser(self, file_path, output_path, placeholder, visualize_ast=False):
        '''
        This is the main function to use the CADTalk parser
        specify the file_path, output_path, placeholder, and visualize_ast
        important: placeholder should at least contain the following format:
        //gt label: ***
        color([0,0,0])
        The following process will utilize the color and *** to locate the line to label
        '''
        success, messages = self.analyser.analyse_file(file_path)
        with open(file_path, 'r') as f:
            code = f.read()
        parse_tree = self.analyser._parse_code(code)
        ast = ast_transformer.transform(parse_tree)
        if visualize_ast:
            ast_graph = ast_visualizer(ast)
            ast_graph.render(os.path.join(os.path.dirname(output_path),f'{os.path.basename(output_path)}_ast'))
        label_lines(file_path, lines_to_label, placeholder, output_path)
        self.assertEqual(success, 1)

    def pass_label(self, labeled_code_path, passing_res_path):
        success, messages = self.analyser.analyse_file(file_path)
        with open(file_path, 'r') as f:
            code = f.read()
        parse_tree = self.analyser._parse_code(code)
        ast = ast_transformer.transform(parse_tree)
        ast_tree = convert_to_tree(ast)
        ast_graph = visualize_ast_use_tree(ast_tree)
        ast_graph.render('ast', view=True)
        with open(labeled_code_path) as f:
            codes = f.readlines()
            gt_labels_raw = [codes[l-2] for l in lines_to_label]
        gt_labels = extract_labels(gt_labels_raw)
        for gt_label, tnode in zip(gt_labels, nodes_to_label):
            tnode.labels.append(gt_label)
        passed_tree = propagate_labels(nodes_to_label)
        ast_graph = visualize_ast_use_tree(passed_tree)
        ast_graph.render('ast', view=True)
        travel_and_get_passed_labels(passed_tree)
        merged_line_and_labels = {}
        for item in line_and_labels:
            line = item['line']
            labels = item['labels']
            if line in merged_line_and_labels:
                merged_line_and_labels[line] = list(set(merged_line_and_labels[line]) | set(labels))
            else:
                merged_line_and_labels[line] = labels
        with open(labeled_code_path) as f:
            code_lines = f.readlines()
        merged_lines_to_label = [(idx, merged_line_and_labels[idx]) for idx in merged_line_and_labels if len(merged_line_and_labels[idx]) > 0]
        for merged_line in reversed(merged_lines_to_label):
            code_lines.insert(merged_line[0]-1, f'//{merged_line[1]}\n')
        with open(passing_res_path, 'w') as f:
            f.writelines(code_lines)
        self.assertEqual(success, 1)

    def pass_lines(self, lines_file):
        success, messages = self.analyser.analyse_file(lines_file)
        with open(lines_file, 'r') as f:
            code = f.read()
        parse_tree = self.analyser._parse_code(code)
        with open(lines_file, 'r') as f:
            codes = f.readlines()
        ast = ast_transformer.transform(parse_tree)
        ast_tree = convert_to_tree(ast)
        ast_graph = visualize_ast_use_tree(ast_tree)
        ast_graph.render('ast', view=True)
        def get_parent_lines(node):
            lines = []
            while node.parent:
                if 'line' in node.data.keys():
                    lines.append(node.data['line'])
                node = node.parent
            return lines
        child_lines = []
        def get_child_lines(node):
            nodeClass = type(node.data).__name__
            if nodeClass == 'dict':
                if 'line' in node.data.keys():
                    child_lines.append(node.data['line'])
            for child in node.children:
                get_child_lines(child)
            return
        def in_def(node):
            while node.parent:
                if 'module_def' in node.data['type']:
                    return True
                node = node.parent
            return False
        def_lines = []
        exec_lines = []
        for node in nodes_to_label:
            child_lines = []
            lines_in_parent = get_parent_lines(node)
            get_child_lines(node)
            all_lines = set(lines_in_parent + child_lines)
            if in_def(node):
                def_lines.append(all_lines)
            else:
                exec_lines.append(all_lines)
        all_def_lines = None
        for lines in def_lines:
            if all_def_lines is None:
                all_def_lines = lines
            else:
                all_def_lines = all_def_lines.union(lines)
        lineGroups = {}
        for line, code in enumerate(codes):
            lineid = line + 1
            linegroup = set()
            for exec_group in exec_lines:
                if lineid in exec_group:
                    linegroup = linegroup.union(exec_group)
            if len(linegroup) > 0:
                lineGroups[lineid] = linegroup
                with open(lines_file, 'r') as f:
                    source_codes = f.readlines()
                filetosave = lines_file.replace('.scad', f'_line{lineid}.scad')
                filetosave_lines = []
                stack = []
                matches = {}
                for line_number, line in enumerate(source_codes):
                    if '{' in line:
                        stack.append(line_number+1)
                    elif '}' in line and stack:
                        opening_line = stack.pop()
                        matches[opening_line] = line_number+1
                tobeCommented_end = []
                tobeCommented_start = []
                difference_lines = []
                for idx, sc in enumerate(source_codes):
                    if idx+1 in linegroup:
                        filetosave_lines.append(sc)
                        continue
                    if '}' in sc:
                        filetosave_lines.append(sc)
                        continue
                    if idx+1 in variable_assignmented_lines:
                        filetosave_lines.append(sc)
                        continue
                    newsc = '//' + sc
                    if idx+1 in matches.keys():
                        tobeCommented_start.append(idx+1)
                        tobeCommented_end.append(matches[idx+1])
                    filetosave_lines.append(newsc)
                for idx in tobeCommented_end:
                    filetosave_lines[idx-1] = '//' +  filetosave_lines[idx-1]
                for vsl in set(variable_assignmented_lines):
                    all_commented = True
                    for sidx in matches.keys():
                        eidx = matches[sidx]
                        if vsl > sidx and vsl < eidx:
                            if not sidx in tobeCommented_start:
                                all_commented = False
                    if all_commented:
                        if vsl-1 > 0 and vsl-1 < len(filetosave_lines):
                            filetosave_lines[vsl-1] = '//' + filetosave_lines[vsl-1]
                with open(filetosave, 'w') as f:
                    f.writelines(filetosave_lines)
        self.assertEqual(success, 1)

    def test_include_import_global(self):
        file_path = os.path.join(self.scadfile_path, 'test_include_import_global.scad')
        success, messages = self.analyser.analyse_file(file_path)
        self.assertEqual(success, 1)
        self.assertEqual(len(messages), 0)

    def test_use_unneeded(self):
        file_path = os.path.join(self.scadfile_path, 'test_use_unneeded.scad')
        success, messages = self.analyser.analyse_file(file_path)
        self.assertEqual(success, 1)
    
    def pass_lines(self, lines_file):
        '''Check that a file with no scad code warns'''
        global line_and_labels
        success, messages = self.analyser.analyse_file(lines_file)
        with open(lines_file,'r') as f:
            code= f.read()
        parse_tree = self.analyser._parse_code(code)
        with open(lines_file,'r') as f:
            codes= f.readlines()
        
        ast = ast_transformer.transform(parse_tree)
        
        ast_tree =convert_to_tree(ast)
        
        ast_graph = visualize_ast_use_tree(ast_tree)
        
        # visualize the AST
        ast_graph.render('ast', view=True) 
        
        def get_parent_lines(node):
            lines = []
            while node.parent:
                if 'line' in node.data.keys():
                    lines.append(node.data['line'])
                node = node.parent
            return lines
        child_lines = []
        def get_child_lines(node):   
            nodeClass= type(node.data).__name__ 
            if nodeClass == 'dict':
                if 'line' in node.data.keys():
                    child_lines.append(node.data['line'])   
            for child in node.children:
                get_child_lines(child)
            return
        def in_def(node):
            while node.parent:
                if 'module_def' in node.data['type']:
                    return True
                node = node.parent
            return False
        def_lines = []
        exec_lines = []
        for node in nodes_to_label:
            child_lines = []
            
            lines_in_parent = get_parent_lines(node)
            get_child_lines(node)
            all_lines = set(lines_in_parent + child_lines)
            if in_def(node):
                def_lines.append(all_lines)
            else:
                exec_lines.append(all_lines)
            # pdb.set_trace()
        all_def_lines = None
        for lines in def_lines:
            if all_def_lines is None:
                all_def_lines = lines
            else:
                all_def_lines = all_def_lines.union(lines)
        lineGroups = {}
        for line, code in enumerate(codes):
            lineid = line + 1
            linegroup = set()
            for exec_group in exec_lines:
                if lineid in exec_group:
                    linegroup = linegroup.union(exec_group)
            if len(linegroup)>0:
                lineGroups[lineid] = linegroup
                #try to save the line file
                with open(lines_file, 'r') as f:
                    source_codes =  f.readlines()
                filetosave = lines_file.replace('.scad', f'_line{lineid}.scad')
                filetosave_lines = []

                # find match for { }
                stack = []
                matches = {}
                # Iterate over lines
                for line_number, line in enumerate(source_codes):
                    # Check for opening bracket
                    if '{' in line:
                        stack.append(line_number+1)
                    # Check for closing bracket
                    elif '}' in line and stack:
                        opening_line = stack.pop()
                        matches[opening_line] =line_number+1
                tobeCommented_end = []
                tobeCommented_start = []
                difference_lines = []
                for idx, sc in enumerate(source_codes):
                    if idx+1 in linegroup:
                        filetosave_lines.append(sc)
                        continue
                    if '}' in sc:
                        filetosave_lines.append(sc)
                        continue
                
                    if idx+1 in variable_assignmented_lines:
                        filetosave_lines.append(sc)
                        continue

                    newsc = '//' + sc
                    if idx+1 in matches.keys():
                        tobeCommented_start.append(idx+1)
                        tobeCommented_end.append(matches[idx+1])
                    filetosave_lines.append(newsc)
                for idx in tobeCommented_end:
                    filetosave_lines[idx-1] = '//' +  filetosave_lines[idx-1]
                for vsl in set(variable_assignmented_lines):
                    all_commented = True
                    for sidx in matches.keys():
                        eidx = matches[sidx]
                        if vsl> sidx and vsl < eidx:
                            
                            if not sidx in tobeCommented_start:
                                all_commented = False
                    if all_commented:
                        print(vsl)
                        if vsl-1>0 and vsl-1<len(filetosave_lines):
                            filetosave_lines[vsl-1] = '//' + filetosave_lines[vsl-1]
                                
                with open(filetosave, 'w')  as f:
                    f.writelines(filetosave_lines)
                # pdb.set_trace()  
        self.assertEqual(success, 1)
    
    def test_include_import_global(self):
        '''Check that a file with no scad code warns'''
        file_path = os.path.join(self.scadfile_path, 'test_include_import_global.scad')
        success, messages = self.analyser.analyse_file(file_path)
        self.assertEqual(success, 1)
        self.assertEqual(len(messages), 0)

    def test_use_unneeded(self):
        '''Check that a file with no scad code warns'''
        file_path = os.path.join(self.scadfile_path, 'test_use_unneeded.scad')
        success, messages = self.analyser.analyse_file(file_path)
        self.assertEqual(success, 1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].code, "W1001")

if __name__ == '__main__':
    # here is the usage example
    parser = FileAnalysisTestCase()
    parser.setUp()
    file_path= 'examples/stage0/input_codes/bike.scad'
    # don't change the placeholder unless you know what is going on
    placeholder = '//gt label: ***\ncolor([0,0,0])'
    output_path = 'examples/stage0/results/bike_with_holder.scad'
    parser.CADTalk_parser(file_path,output_path,placeholder,visualize_ast=True)

