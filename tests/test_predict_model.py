from click.testing import CliRunner

from src import predict_model

# Тест, что метод исполняется и CLI работает:
runner = CliRunner()


def test_cli_command():
    result = runner.invoke(predict_model, "data/interim/train_postproc.csv")
    assert result.exit_code == 0
