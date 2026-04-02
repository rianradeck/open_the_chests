from openthechests.src.elements.Parser import Parser
from openthechests.src.elements.Generator import Generator
from openthechests.src.elements.Pattern import Pattern
from open_the_chests.envs.otc_registry import (
    all_event_types, all_event_attributes,
    all_noise_types, all_noise_attributes,
    instructions_easy, instructions_medium, instructions_hard
)

INSTRUCTIONS = {
    "easy":   instructions_easy,
    "medium": instructions_medium,
    "hard":   instructions_hard,
}

_generators: dict = {}


def _get_generator(env: str) -> Generator:
    if env not in _generators:
        instructions = INSTRUCTIONS[env]
        parser   = Parser(all_event_types, all_noise_types, all_event_attributes, all_noise_attributes)
        patterns = [Pattern(instr, i) for i, instr in enumerate(instructions)]
        _generators[env] = Generator(parser, patterns)
    return _generators[env]


def generate_sequence(n_events=200, env: str = "medium"):
    generator = _get_generator(env)
    generator.reset()
    events = []
    signals_list = []
    for _ in range(n_events):
        event, signals = generator.next_event()
        events.append(event)
        signals_list.append(signals)
    return events, signals_list


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openthechests.src.utils.modified_plotting import draw_event_sequence_matplot

    events, signals = generate_sequence(n_events=5)
    draw_event_sequence_matplot(events, env_name="Easy")
    plt.savefig("sequence.png")
    print("Saved sequence.png")
