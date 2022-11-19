from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from parse_html import *
import pickle
import os

tasklist = ['Task 1', 'Task 2', 'Task 3']
mydict = dict()

class requestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("content-type", "text/html")
        # self.send_header('Access-Control-Allow-Origin', '*')                
        # self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        # self.send_header("Access-Control-Allow-Headers", "X-Requested-With")       
        # self.send_header("content-type", "text/html")
        # self.send_header("Access-Control-Allow-Headers", "Access-Control-Allow-Origin, Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers")
        self.end_headers()
        global mydict
        if self.path.endswith("sidebar"):
            with open("sidebar.html", "r") as f:
                output = f.read()
        elif self.path.endswith("notes"):
            # with open("notes.html", "r") as f:
            #     output = f.read()
            output = '''<h1>A very naive note space</h1>
                        <div id="all_notes">
                        '''
            for i in mydict:
                for j in mydict[i]:
                    for kindex , k in enumerate(mydict[i][j]):
                        if kindex % 2 == 1:
                            output += "<p>From question {} answer {}, you labeled: {}.</p>".format(i, j, k)
            output += "</div>"
        else:
            last = self.path.split("/")[-1]
            if last in mydict:
                output = mydict[last]
            else:
                output = {}
            output = json.dumps(output)
        self.wfile.write(output.encode())

    def do_POST(self):        
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")       
        self.send_header("content-type", "text/html")
        self.send_header("Access-Control-Allow-Headers", "Access-Control-Allow-Origin, Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers")
        self.end_headers()

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        arguments = post_data.decode('utf-8').split("bonankou")

        if arguments[0] == "summary":
            question = arguments[2]
            answer = arguments[1]
            answer_id = arguments[3]
            # fake_question = "What is a non-capturing group in regular expressions? How are non-capturing groups, i.e. (?:), used in regular expressions and what are they good for?"
            output, best_predictions, code_exist = parse(question = question, answers = [answer], answerIds=[answer_id])
            self.wfile.write((output[0] + "bonankou" + best_predictions[0] + "bonankou" + code_exist[0]).encode())
        elif arguments[0] == "update":
            global mydict
            newcomer = json.loads(arguments[1])
            if newcomer != {}:
                mydict[list(newcomer.keys())[0]] = list(newcomer.values())[0]
            else:
                mydict[list(newcomer.keys())[0]] = {}
            pickle.dump(mydict, open("mydict.txt", "wb"))

    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")       
        self.send_header("content-type", "text/html")
        self.send_header("Access-Control-Allow-Headers", "Access-Control-Allow-Origin, Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers")
        self.end_headers()


def main():
    PORT = 8000
    server = HTTPServer(('', PORT), requestHandler)
    print("Server running on port {}".format(PORT))
    global mydict
    if os.path.isfile("mydict.txt"):
        mydict = pickle.load(open("mydict.txt", "rb"))
    server.serve_forever()


if __name__ == "__main__":
    main()