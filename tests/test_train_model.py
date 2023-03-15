from click.testing import CliRunner

from src import train_model

# Тест, что метод исполняется и CLI работает:
runner = CliRunner()


def test_cli_command():
    result = runner.invoke(
        train_model, "data/interim/train_postproc.csv models/model.clf"
    )
    assert result.exit_code == 0
