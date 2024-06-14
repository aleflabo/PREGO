import heapq
import random
import warnings

import numpy as np
import torch


def dijkstra_shortest_path(graph, start, end, search_nodes="all"):
    # Check if start node exists in the graph
    if start not in graph:
        return [], float("inf")
    # If end node is not in the graph, add it with edges from its connected nodes with infinite weight
    if end not in graph:
        graph[end] = {node: float("inf") for node in graph.keys() if end in graph[node]}

    assert search_nodes == "all" or isinstance(search_nodes, list)
    # Initialize distances and previous nodes
    if search_nodes == "all":
        distances = {node: float("inf") for node in graph}
        previous_nodes = {node: None for node in graph}
    else:
        if start not in search_nodes:
            search_nodes.append(start)
        if end not in search_nodes:
            search_nodes.append(end)
        distances = {node: float("inf") for node in graph if node in search_nodes}
        previous_nodes = {node: None for node in graph if node in search_nodes}
    distances[start] = 0

    # Priority queue to store nodes to visit
    priority_queue = [(0, start)]

    while priority_queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # Stop if we reach the end node
        if current_node == end:
            break

        # Update distances for neighboring nodes
        if current_node not in graph:
            continue
        for neighbor, weight in graph[current_node].items():
            if search_nodes != "all" and neighbor not in search_nodes:
                continue
            distance = current_distance + weight
            if neighbor not in distances:
                distances[neighbor] = float("inf")
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Trace back the shortest path from end to start
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]

    # Reverse the path and return it as a list
    path.reverse()
    return path, distances[end]


class TaskGraph:
    def __init__(
        self,
        dataset="coin",
        graph_type="overall",
        graph_modality="video",
        graph_fusion="weighted-average",
        alpha=0.5,
        merge_video_text=False,
    ):
        # Init
        assert graph_type in ["overall", "task"]  # Type of task graph
        assert graph_modality in ["video", "text", "video-text"]
        self.graph_type = graph_type
        self.graph_modality = graph_modality
        # If graph_type is task specific, then dict format will be
        # graph = {'class': 'graph'}
        if graph_modality != "video-text":
            self.task_graph = {graph_modality: {}}
        else:
            self.task_graph = {"video": {}, "text": {}}
            self.merge_video_text = merge_video_text
            if merge_video_text:
                self.graph_fusion = graph_fusion
                self.alpha = (
                    alpha  # For weighted average of graph fusion (alpha=video weight)
                )
        self.finalized = False  # Finalize task graph before using it for eval
        self.sentence_database = {}  # List all keysteps OR keysteps per task
        self.all_scores = []  # To find max and min
        # Right now only for COIN
        self.dataset = dataset
        self.find_statistics()

    def register_sequence(
        self,
        feat_embeds=None,
        keystep_embeds=None,
        keystep_sents=None,
        sim=None,
        input_modality=None,
        class_idx=None,
    ):
        """
        feat_embeds in the feature sequence embeddings
        keystep_embeds are the candidate keystep embeddings
        keystep_sents are the raw sentences for setting dict keys
        """
        assert (feat_embeds is not None and keystep_embeds is not None) or (
            sim is not None
        )
        if not input_modality and self.graph_modality == "video-text":
            assert False, "You need to specify modality in video-text task_graph"
        assert self.graph_type == "overall" or (
            self.graph_type == "task" and class_idx is not None
        )
        if self.graph_type == "task":
            # First populate the database
            if int(class_idx) not in self.sentence_database:
                self.sentence_database[int(class_idx)] = keystep_sents
            else:
                assert keystep_sents == self.sentence_database[int(class_idx)] and len(
                    keystep_sents
                ) == len(self.sentence_database[int(class_idx)])
            # Append class_idx and @@@ symbol in the front.
            keystep_sents = [
                str(class_idx) + "@@@" + x.replace("@@@", " ") for x in keystep_sents
            ]
        else:
            if self.sentence_database == {}:
                self.sentence_database = keystep_sents
            else:
                assert keystep_sents == self.sentence_database and len(
                    keystep_sents
                ) == len(self.sentence_database)
        if self.graph_modality != "video-text":
            if input_modality is not None:
                # if input_modality is set, it must be same as graph_modality if not video-text
                assert self.graph_modality == input_modality
            else:
                input_modality = self.graph_modality
        try:
            assert keystep_embeds.shape[0] == len(keystep_sents)
        except:
            assert sim.shape[1] == len(keystep_sents)
        # if only one feature then we cannot register
        if (feat_embeds is not None and len(feat_embeds) == 1) or (
            sim is not None and sim.shape[0] == 1
        ):
            return
        if isinstance(feat_embeds, np.ndarray):
            feat_embeds = torch.from_numpy(feat_embeds)
        if isinstance(keystep_embeds, np.ndarray):
            keystep_embeds = torch.from_numpy(keystep_embeds)
        if isinstance(sim, np.ndarray):
            sim = torch.from_numpy(sim)
        if self.graph_type == "task":
            if int(class_idx) not in self.sentence_database:
                self.sentence_database[int(class_idx)] = keystep_sents

        # Input the sequence and update the graph
        if sim is None:
            sim = feat_embeds @ keystep_embeds.T

        best_keysteps = torch.argmax(sim, dim=1)
        # print(best_keysteps.shape)
        # print(best_keysteps)
        for time_idx in range(1, len(best_keysteps)):
            source = keystep_sents[int(best_keysteps[time_idx - 1])]
            dest = keystep_sents[int(best_keysteps[time_idx])]
            if source not in self.task_graph[input_modality]:
                self.task_graph[input_modality][source] = {dest: 1}
            if dest not in self.task_graph[input_modality][source]:
                self.task_graph[input_modality][source][dest] = 1
            else:
                self.task_graph[input_modality][source][dest] += 1

    def normalize_graph(self, graph):
        """
        Normalize the edge weights of a graph so that the sum of outward edges from every node sums to 1.

        Parameters:
        graph (dict): The graph stored as a dictionary.

        Returns:
        dict: The normalized graph.
        """
        normalized_graph = {}

        # Compute the sum of outward edge weights for each node
        for node, edges in graph.items():
            sum_weights = sum(edges.values())

            # Normalize the edge weights for the node
            if sum_weights > 0:
                normalized_edges = {
                    neighbor: weight / sum_weights for neighbor, weight in edges.items()
                }
                normalized_graph[node] = normalized_edges

        return normalized_graph

    def merge_graphs(self, video_graph, text_graph, alpha):
        """
        Merge two graphs using a weighted average.

        Parameters:
        video_graph (dict): The video graph stored as a dictionary.
        text_graph (dict): The text graph stored as a dictionary.
        alpha (float): The weight of the video graph.

        Returns:
        dict: The merged graph.
        """
        # Normalize the edge weights of both graphs
        video_graph = self.normalize_graph(video_graph)
        text_graph = self.normalize_graph(text_graph)

        fused_graph = {}

        # Merge nodes from both graphs
        nodes = set(video_graph.keys()) | set(text_graph.keys())

        # Compute weighted average of edge weights for each node
        for node in nodes:
            # Compute weighted average of outgoing edges
            outgoing_edges = {}
            if node in video_graph:
                for neighbor, weight in video_graph[node].items():
                    outgoing_edges[neighbor] = alpha * weight
            if node in text_graph:
                for neighbor, weight in text_graph[node].items():
                    outgoing_edges[neighbor] = (
                        outgoing_edges.get(neighbor, 0) + (1 - alpha) * weight
                    )

            # Normalize outgoing edge weights
            fused_graph[node] = self.normalize_graph({node: outgoing_edges})[node]

        return fused_graph

    def check_and_finalize(self):
        if not self.finalized:
            warnings.warn(
                "Task Graph must be finalized for normalization and merging... Doing that now..."
            )
            self.finalize_task_graph()

    def find_statistics(self, tail_prob=0):
        """
        Some methods have scores outside [0, 1] we need to re-scale the numbers
        To achieve this, run the statistics when updating task graph and rescale when eval
        Set everything beyond tail_prob to 0.0 and more than 1-tail_prob to 1.0
        """
        coin_videoclip_statistics = {
            0: -23.828975677490234,
            5: -3.1052308082580566,
            10: -1.9405536651611328,
            15: -1.205209732055664,
            20: -0.6483414173126221,
            25: -0.18889069557189941,
            30: 0.21026432514190674,
            35: 0.5699708461761475,
            40: 0.902841329574585,
            45: 1.2174692153930664,
            50: 1.5206257104873657,
            55: 1.8181183338165283,
            60: 2.1148629188537598,
            65: 2.4166736602783203,
            70: 2.7299509048461914,
            75: 3.0637478828430176,
            80: 3.432137966156006,
            85: 3.8598978519439697,
            90: 4.401253700256348,
            95: 5.230754852294922,
            100: 24.11414337158203,
        }
        crosstask_mil_statistics = {
            0: -30.90512466430664,
            5: -3.146651554107666,
            10: -1.8409147024154662,
            15: -0.9835262954235077,
            20: -0.28699212670326224,
            25: 0.3074645400047302,
            30: 0.8429063081741333,
            35: 1.3513998329639434,
            40: 1.8323472976684572,
            45: 2.296899402141571,
            50: 2.762255907058716,
            55: 3.2211565971374516,
            60: 3.68434157371521,
            65: 4.160336136817933,
            70: 4.662180995941162,
            75: 5.199650526046753,
            80: 5.796158981323243,
            85: 6.467047238349915,
            90: 7.277130603790284,
            95: 8.412010717391968,
            100: 16.75782585144043,
        }
        if self.dataset == "coin":
            assert (
                tail_prob in coin_videoclip_statistics
                and 100 - tail_prob in coin_videoclip_statistics
            )
            self.clamp_max = coin_videoclip_statistics[100 - tail_prob]
            self.clamp_min = coin_videoclip_statistics[tail_prob]
            return
        elif self.dataset == "crosstask":
            assert (
                tail_prob in crosstask_mil_statistics
                and 100 - tail_prob in crosstask_mil_statistics
            )
            self.clamp_max = crosstask_mil_statistics[100 - tail_prob]
            self.clamp_min = crosstask_mil_statistics[tail_prob]
            return
        print("Evaluating score statistics to re-scale the scores to [0, 1]...")
        # Find the maximum and minimum values
        max_val = min_val = self.all_scores[0]
        for num in self.all_scores[1:]:
            if num > max_val:
                max_val = num
            elif num < min_val:
                min_val = num

        # Define the quickselect function
        def quickselect(lst, k):
            pivot = random.choice(lst)
            lows = [el for el in lst if el < pivot]
            highs = [el for el in lst if el > pivot]
            pivots = [el for el in lst if el == pivot]
            if k < len(lows):
                return quickselect(lows, k)
            elif k < len(lows) + len(pivots):
                return pivots[0]
            else:
                return quickselect(highs, k - len(lows) - len(pivots))

        # Find the percentiles using quickselect
        n = len(self.all_scores)
        percentile_low = quickselect(self.all_scores, int(tail_prob * n))
        percentile_high = quickselect(self.all_scores, int((1 - tail_prob) * n) - 1)

        self.max_val = max_val
        self.min_val = min_val
        self.clamp_max = percentile_high
        self.clamp_min = percentile_low

        print("Clamp max: {} and clamp min: {}".format(self.clamp_max, self.clamp_min))

        # print(self.max_val, self.min_val, self.clamp_max, self.clamp_min)

    def finalize_task_graph(self):
        """
        Take average
        Merge task graphs if video-text
        """
        if self.graph_modality != "video-text":
            self.task_graph = self.normalize_graph(self.task_graph[self.graph_modality])
        elif self.merge_video_text:
            # Merge video and text graphs
            self.task_graph = self.merge_graphs(
                self.task_graph["video"], self.task_graph["text"], self.alpha
            )
        else:
            # If not merging then we are ready to use task graph directly
            pass
        # If graph_type is task, we need to split the graph
        if self.graph_type == "task":
            # When registering we make sure to prepend class_idx in the beggining, now extract that
            per_task_graph = {}
            for key in self.task_graph:
                class_idx, _ = key.split("@@@")
                if int(class_idx) not in per_task_graph:
                    per_task_graph[int(class_idx)] = {}
                per_task_graph[int(class_idx)][key] = self.task_graph[key]
            self.task_graph = per_task_graph
        self.finalized = True

    def visualize_task_graph(self):
        self.check_and_finalize()
        if self.graph_type == "task":
            for class_idx in self.task_graph:
                print(
                    "For class idx: {}, printing some transitions...".format(class_idx)
                )
                class_task_graph = self.task_graph[class_idx]
                for node, edges in class_task_graph.items():
                    sorted_edges = sorted(edges, key=lambda x: edges[x], reverse=True)[
                        :3
                    ]
                    print(f"Node {node}: Top 3 highest next nodes: {sorted_edges}")
                print("#" * 100)

    def build_adjacency_matrix(self):
        self.check_and_finalize()
        # (TODO) ashutoshkr: might not work with unfused task graphs
        # sentence_database can give us the matrix size
        if self.graph_type == "task":
            self.adjacency_matrix = {}
            for class_idx in self.task_graph:
                matrix_dim = len(self.sentence_database[class_idx])
                self.adjacency_matrix[class_idx] = torch.zeros((matrix_dim, matrix_dim))
                for source in self.task_graph[class_idx]:
                    for dest in self.task_graph[class_idx][source]:
                        source_idx = self.sentence_database[class_idx].index(
                            source.split("@@@")[1]
                        )
                        dest_idx = self.sentence_database[class_idx].index(
                            dest.split("@@@")[1]
                        )
                        assert source_idx != -1 and dest_idx != -1
                        self.adjacency_matrix[class_idx][source_idx][
                            dest_idx
                        ] = self.task_graph[class_idx][source][dest]
        else:
            matrix_dim = len(self.sentence_database)
            self.adjacency_matrix = torch.zeros(matrix_dim, matrix_dim)
            for source in self.task_graph:
                for dest in self.task_graph[source]:
                    source_idx = self.sentence_database.index(source)
                    dest_idx = self.sentence_database.index(dest)
                    assert source_idx != -1 and dest_idx != -1
                    self.adjacency_matrix[source_idx][dest_idx] = self.task_graph[
                        source
                    ][dest]

    def beam_search(self, start_state, end_state, transition_matrix, beam_width, k):
        """
        Beam search algorithm to find the top-k paths from start state to end state with maximum probability.

        Args:
            start_state (hashable): The starting state.
            end_state (hashable): The end state.
            transition_matrix (dict): A dictionary representing the state transition matrix.
                                    Keys are current states, and values are dictionaries
                                    representing next states and their transition probabilities.
            beam_width (int): The beam width to use in the search.
            k (int): The number of top-k paths to return.

        Returns:
            A list of tuples, each containing a path from start state to end state and its probability.
        """
        # Initialize the beam with the start state and its probability
        beam = [(0.0, [start_state])]

        # Initialize a list to store the top-k paths found so far
        top_k = []

        # Initialize a set to store the visited states in the current path
        visited = set()

        # Iterate until the beam reaches the end state or is empty
        num_retries = 50
        retry_count = 0
        while beam and retry_count < num_retries:
            # print(beam)
            # xx = input()
            # Get the next level of states by expanding the states in the beam
            next_level = []
            for probability, path in beam:
                current_state = path[-1]
                visited.add(current_state)
                if current_state == end_state:
                    # If the end state is reached, add the path and its probability to the list of top-k paths
                    top_k.append((probability, path))
                    # print('()'*100)
                else:
                    for next_state, transition_prob in transition_matrix[
                        current_state
                    ].items():
                        if (
                            next_state not in visited
                        ):  # Ignore visited states in the current path
                            next_path = path + [next_state]
                            next_prob = probability + transition_prob
                            next_level.append((next_prob, next_path))

            # Keep only the top beam_width paths and discard the rest
            beam = heapq.nlargest(beam_width, next_level)

            # Clear the visited set for the next iteration
            visited.clear()
            retry_count += 1

        # Sort the top-k paths by their probabilities and return the top-k paths
        print(top_k)
        return sorted(top_k, reverse=True)[:k]

    def predict_keystep(
        self,
        feat_embeds=None,
        keystep_embeds=None,
        sim=None,
        keystep_sents=None,
        input_modality=None,
        class_idx=None,
        method="baseline",
        discard_percentile=35,
    ):
        """
        Think about how to fuse video and text measurements
        NOTE: It will give results out of keystep_embeds option
        """
        self.check_and_finalize()
        assert self.graph_type == "overall" or (
            self.graph_type == "task" and class_idx is not None
        )
        if isinstance(feat_embeds, np.ndarray):
            feat_embeds = torch.from_numpy(feat_embeds)
        if isinstance(keystep_embeds, np.ndarray):
            keystep_embeds = torch.from_numpy(keystep_embeds)
        if method == "baseline":
            # Add threshold
            if not hasattr(self, "clamp_max"):
                self.find_statistics()
            if sim is None:
                sim = feat_embeds @ keystep_embeds.T
            # Transform the score to [0, 1]
            sim = sim.clamp(self.clamp_min, self.clamp_max)
            assert self.clamp_max - self.clamp_min > 1e-3
            sim = (sim - self.clamp_min) / (self.clamp_max - self.clamp_min)
            max_scores, preds = torch.max(sim, dim=1)
            thresh = np.percentile(max_scores.numpy(), discard_percentile)
            preds[max_scores < thresh] = -1
            # Apply threshold
            # random_thresh = 0.7
            # preds[max_scores < random_thresh] = random.randint(0, sim.shape[1])
            return preds.numpy(), 0  # sum(max_scores < random_thresh)

        elif method == "bayes-filter":
            if not hasattr(self, "adjacency_matrix"):
                self.build_adjacency_matrix()
            if not hasattr(self, "clamp_max"):
                self.find_statistics()
            if sim is None:
                sim = feat_embeds @ keystep_embeds.T
            # Transform the score to [0, 1]
            sim = sim.clamp(self.clamp_min, self.clamp_max)
            assert self.clamp_max - self.clamp_min > 1e-3
            sim = (sim - self.clamp_min) / (self.clamp_max - self.clamp_min)

            labels = np.zeros(sim.shape[0])
            max_scores = np.zeros(sim.shape[0])
            for time_idx in range(sim.shape[0]):
                if time_idx == 0:
                    belief = sim[time_idx]
                else:
                    if self.graph_type == "overall":
                        adj_mat = self.adjacency_matrix
                    else:
                        adj_mat = self.adjacency_matrix[class_idx]
                    belief = sim[time_idx] + 0.20 * torch.mv(
                        adj_mat, belief / np.linalg.norm(belief, 1)
                    )
                assert np.linalg.norm(belief, 2) > 0
                # belief = belief/4.0
                sim[time_idx] = belief
                # belief = belief/np.linalg.norm(belief, 2)
                labels[time_idx] = int(torch.argmax(belief))
                max_scores[time_idx] = float(belief[int(torch.argmax(belief))])
            max_scores, preds = torch.max(sim, dim=1)
            thresh = np.percentile(max_scores.numpy(), discard_percentile)
            preds[max_scores < thresh] = -1
            # Apply threshold
            # random_thresh = 0.7
            # preds[max_scores < random_thresh] = random.randint(0, sim.shape[1])
            return preds.numpy(), 0
            thresh = np.percentile(max_scores, discard_percentile)
            labels[max_scores < thresh] = -1
            # print(sum((labels == -1))/len(labels))
            return labels, 0

        elif method == "beam-search":
            # Add threshold
            if not hasattr(self, "clamp_max"):
                self.find_statistics()
            if sim is None:
                sim = feat_embeds @ keystep_embeds.T
            # Transform the score to [0, 1]
            sim = sim.clamp(self.clamp_min, self.clamp_max)
            assert self.clamp_max - self.clamp_min > 1e-3
            sim = (sim - self.clamp_min) / (self.clamp_max - self.clamp_min)
            max_scores, preds = torch.max(sim, dim=1)

            mask = max_scores < 0.77
            new_preds = preds.clone()
            new_preds[mask] = -100
            print(new_preds)
            print(preds)
            exit()

            pivots = preds[max_scores > 0.8]
            print(pivots)
            print(keystep_sents)
            print(torch.max(max_scores), torch.min(max_scores))
            thresh = np.percentile(max_scores.numpy(), discard_percentile)
            preds[max_scores < thresh] = -1
            if self.graph_type == "task":
                curr_task_graph = self.task_graph[class_idx]
            print(curr_task_graph.keys())
            print(curr_task_graph)
            # exit()
            top_k = self.beam_search(
                "44789@@@Pop the hood of the car",
                "44789@@@Pull out the dipstick and wipe it down with a dry cloth",
                curr_task_graph,
                beam_width=5,
                k=1,
            )
            # '44789@@@Find the oil fill port'
            print("$$$$$$$$$$$$$$$$$$")
            print(top_k)
            exit()
            # Apply threshold
            # random_thresh = 0.7
            # preds[max_scores < random_thresh] = random.randint(0, sim.shape[1])
            print(preds.shape)
            return preds.numpy(), 0  # sum(max_scores < random_thresh)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    graph = TaskGraph()
    feat_embeds = torch.rand(100, 512)
    keystep_embeds = torch.rand(12, 512)
    keystep_sents = [
        "The cat slept soundly in the sunbeam.",
        "She danced through the meadow, her dress fluttering in the breeze.",
        "The rain fell in sheets, drenching the city streets.",
        "He spoke with authority, commanding the attention of the room.",
        "The mountain loomed in the distance, shrouded in mist.",
        "The orchestra played a haunting melody, sending shivers down her spine.",
        "The wind howled through the trees, signaling an approaching storm.",
        "The old man sat quietly on the park bench, lost in thought.",
        "The waves crashed against the shore, leaving behind a trail of foam.",
        "The little boy chased after the butterfly, his eyes alight with wonder.",
        "The smell of freshly baked bread wafted through the air, making her stomach rumble.",
        "The stars twinkled in the night sky, casting a soft glow over the landscape.",
    ]
    graph.register_sequence(feat_embeds, keystep_embeds, keystep_sents)
    # print(keystep_embeds.shape)
    exit()

    feat_embeds, keystep_embeds, keystep_sents
