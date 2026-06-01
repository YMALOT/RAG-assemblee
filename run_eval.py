
import argparse

from evaluation import load_questions, evaluate, QUESTIONS_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-procedural", action="store_true",
                    help="exclude procedural chunks during retrieval")
    ap.add_argument("--rerank", action="store_true",
                    help="use cross-encoder reranking on top of dense retrieval")
    ap.add_argument("--hybrid", action="store_true",
                    help="use hybrid dense+BM25 retrieval with RRF fusion")
    args = ap.parse_args()
 
    questions = load_questions(args.questions)
    print(f"Evaluating {len(questions)} questions (k={args.k})\n" + "=" * 70)
    if args.rerank and args.hybrid:
        raise SystemExit("Choose --rerank OR --hybrid, not both.")
    if args.rerank:
        from generation import RerankedRetriever
        retriever = RerankedRetriever()
    elif args.hybrid:
        from generation import HybridRetriever
        retriever = HybridRetriever()
    else:
        from generation import Retriever
        retriever = Retriever()
    evaluate(retriever, questions, k=args.k, exclude_procedural=args.no_procedural)


if __name__ == "__main__":
    main()