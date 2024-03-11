import unittest
import tempfile
import subprocess
import pexpect
import re


def _remove_ansi_escape_sequences(text) -> str:
    ansi_escape_pattern = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape_pattern.sub('', text)


def SMCDEL(text: str) -> bool:
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(text.encode())
        temp.seek(0)
        tmp_path = temp.name

        process = subprocess.run(
            ["smcdel", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            raise ValueError("smcdel failed to execute")

        output = str(process.stdout)
        output = output.split("\n")
        output = output[-5]
        output = _remove_ansi_escape_sequences(output)
        output = output.strip()
        output = True if output == "True" else False
        return output


class TestSMCDEL(unittest.TestCase):
    def test_SMCDEL_True(self):
        result = SMCDEL(
            text="VARS 1,2,3,4 LAW Top OBS Agenta:1 Agentb:2 Agentc:3 Agentd:4 VALID? [ ! (1|2|3|4) ] [ ! ~(Agenta knows whether (1&2&3&4)) ] [ ! ~(Agenta knows whether 2) ] Agentd knows whether (Agenta knows whether 3)")
        self.assertEqual(result, True)

    def test_SMCDEL_False(self):
        result = SMCDEL(
            text="VARS 1,2,3,4 LAW Top OBS Agenta:1 Agentb:2 Agentc:3 Agentd:4 VALID? [ ! (1|2|3|4) ] [ ! ~(Agentd knows whether (1&2&3&4)) ] [ ! ~(Agentd knows whether 2) ] Agenta knows that (Agentb knows whether (1&2&3&4))")
        self.assertEqual(result, False)

    def test_SMCDEL_Error(self):
        with self.assertRaises(ValueError):
            SMCDEL(text="wrong input")


if __name__ == '__main__':
    unittest.main()
