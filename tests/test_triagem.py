import io
import json
import unittest
from contextlib import redirect_stdout

from triagem.avaliacao import (
    avaliar,
    curva_abstencao,
    dobras,
    erros_mais_confiantes,
    matriz_confusao,
    relatorio,
    validacao_cruzada,
)
from triagem.cli import RAIZ, main
from triagem.dados import CLASSES, dividir, gerar
from triagem.modelos import Logistica, Majoritario, NaiveBayes, Regras
from triagem.texto import normalizar, tokens

BRINQUEDO = [
    ("quero ver a fatura do cartao", "cobranca"),
    ("fui cobrado duas vezes no cartao", "cobranca"),
    ("o boleto da fatura veio errado", "cobranca"),
    ("esqueci a senha da conta", "acesso"),
    ("nao consigo entrar na conta com a senha", "acesso"),
    ("senha bloqueada na conta", "acesso"),
] * 3


class DadosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dados = gerar(750)

    def test_e_deterministico(self):
        self.assertEqual(self.dados, gerar(750))
        self.assertNotEqual(self.dados, gerar(750, semente=1))

    def test_textos_unicos_e_balanceados(self):
        textos = [t for t, _ in self.dados]
        self.assertEqual(len(textos), len(set(textos)))
        for c in CLASSES:
            self.assertEqual(sum(1 for _, y in self.dados if y == c), 150)

    def test_divisao_sem_vazamento_e_estratificada(self):
        treino, teste = dividir(self.dados)
        self.assertEqual({t for t, _ in treino} & {t for t, _ in teste}, set())
        self.assertEqual(len(treino) + len(teste), 750)
        for c in CLASSES:
            self.assertEqual(sum(1 for _, y in teste if y == c), 45)

    def test_arquivo_gravado_bate_com_o_gerador(self):
        linhas = (RAIZ / "data" / "chamados.jsonl").read_text(encoding="utf-8").splitlines()
        gravado = [(d["texto"], d["classe"]) for d in map(json.loads, linhas)]
        self.assertEqual(gravado, self.dados)

    def test_n_menor_que_o_numero_de_classes_gera_vazio(self):
        self.assertEqual(gerar(4), [])

    def test_o_espaco_de_textos_comporta_conjuntos_maiores(self):
        grande = gerar(5000)
        self.assertEqual(len({t for t, _ in grande}), 5000)


class TextoTest(unittest.TestCase):
    def test_normaliza_e_gera_bigramas(self):
        self.assertEqual(normalizar("Cobrança"), "cobranca")
        self.assertEqual(tokens("Não chegou", bigramas=True), ["nao", "chegou", "nao_chegou"])
        self.assertEqual(tokens("Não chegou", bigramas=False), ["nao", "chegou"])


class ModelosTest(unittest.TestCase):
    def setUp(self):
        self.x = [t for t, _ in BRINQUEDO]
        self.y = [c for _, c in BRINQUEDO]

    def test_probabilidades_somam_um(self):
        for m in (Majoritario(), Regras(), NaiveBayes(), Logistica(epocas=5)):
            m.fit(self.x, self.y)
            self.assertAlmostEqual(sum(m.proba("cartao fatura senha").values()), 1.0, places=6, msg=m.nome)

    def test_naive_bayes_aprende_o_brinquedo(self):
        m = NaiveBayes().fit(self.x, self.y)
        self.assertEqual(m.prever("a fatura do cartao"), "cobranca")
        self.assertEqual(m.prever("senha da conta"), "acesso")

    def test_logistica_aprende_o_brinquedo_e_e_deterministica(self):
        a = Logistica(epocas=10).fit(self.x, self.y)
        b = Logistica(epocas=10).fit(self.x, self.y)
        self.assertEqual(a.prever("cobrado no cartao"), "cobranca")
        self.assertEqual(a.proba("fatura"), b.proba("fatura"))
        self.assertIn("senha", [t for t, _ in a.palavras_mais_fortes("acesso", 5)])

    def test_palavra_desconhecida_nao_quebra(self):
        m = NaiveBayes().fit(self.x, self.y)
        self.assertEqual(round(sum(m.proba("zzzz qqqq").values()), 6), 1.0)

    def test_regras_sem_palavra_conhecida_fica_indeciso(self):
        m = Regras().fit(self.x, self.y)
        p = m.proba("blablabla")
        self.assertAlmostEqual(p["acesso"], 0.5)

    def test_majoritario_devolve_a_classe_mais_frequente(self):
        m = Majoritario().fit(["a", "b", "c"], ["x", "x", "y"])
        self.assertEqual(m.prever("qualquer"), "x")


class MetricasTest(unittest.TestCase):
    def test_relatorio_com_valores_calculados_a_mao(self):
        reais = ["a", "a", "a", "b", "b"]
        previstos = ["a", "a", "b", "b", "a"]
        r = relatorio(reais, previstos, ["a", "b"])
        self.assertAlmostEqual(r["acuracia"], 3 / 5)
        # classe a: tp=2 fp=1 fn=1 -> p=r=2/3 ; classe b: tp=1 fp=1 fn=1 -> p=r=1/2
        self.assertAlmostEqual(r["por_classe"]["a"]["f1"], 2 / 3)
        self.assertAlmostEqual(r["por_classe"]["b"]["f1"], 1 / 2)
        self.assertAlmostEqual(r["f1_macro"], (2 / 3 + 1 / 2) / 2)

    def test_matriz_de_confusao(self):
        m = matriz_confusao(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(m, {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 1}})

    def test_dobras_sao_disjuntas_e_cobrem_tudo(self):
        d = dobras(23, 5)
        todos = [i for dobra in d for i in dobra]
        self.assertEqual(sorted(todos), list(range(23)))
        self.assertEqual(len(set(todos)), 23)

    def test_validacao_cruzada_do_brinquedo(self):
        cv = validacao_cruzada(lambda: NaiveBayes(), BRINQUEDO, 3)
        self.assertGreater(cv["f1_macro"][0], 0.8)
        self.assertEqual(cv["dobras"], 3)


class AbstencaoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dados = gerar(250)
        cls.treino, cls.teste = dividir(dados)
        cls.modelo = NaiveBayes().fit([t for t, _ in cls.treino], [y for _, y in cls.treino])

    def test_cobertura_nao_aumenta_com_o_limiar(self):
        linhas = curva_abstencao(self.modelo, self.teste, [0.0, 0.5, 0.7, 0.9, 0.999])
        coberturas = [l["cobertura"] for l in linhas]
        self.assertEqual(coberturas, sorted(coberturas, reverse=True))
        self.assertEqual(linhas[0]["cobertura"], 1.0)
        self.assertEqual(linhas[0]["para_humano"], 0)

    def test_erros_vem_ordenados_por_confianca(self):
        erros = erros_mais_confiantes(self.modelo, self.teste, 5)
        confs = [e["confianca"] for e in erros]
        self.assertEqual(confs, sorted(confs, reverse=True))
        self.assertTrue(all(e["real"] != e["previsto"] for e in erros))


class DesempenhoTest(unittest.TestCase):
    """Garante que o resultado publicado no README continua verdadeiro."""

    @classmethod
    def setUpClass(cls):
        cls.dados = gerar(750)
        cls.treino, cls.teste = dividir(cls.dados)
        cls.x, cls.y = [t for t, _ in cls.treino], [y for _, y in cls.treino]

    def test_modelos_aprendidos_superam_a_linha_de_base(self):
        base = avaliar(Majoritario().fit(self.x, self.y), self.teste)["f1_macro"]
        nb = avaliar(NaiveBayes().fit(self.x, self.y), self.teste)["f1_macro"]
        lg = avaliar(Logistica().fit(self.x, self.y), self.teste)["f1_macro"]
        self.assertLess(base, 0.1)
        self.assertGreater(nb, 0.8)
        self.assertGreater(lg, 0.85)


class CliTest(unittest.TestCase):
    def rodar(self, *args):
        saida = io.StringIO()
        with redirect_stdout(saida):
            codigo = main(list(args) + ["--n", "250"])
        return codigo, saida.getvalue()

    def test_prever_encaminha_mensagem_ambigua_para_humano(self):
        _, texto = self.rodar("prever", "blablabla", "--limiar", "0.9")
        self.assertIn("encaminhar para um humano", texto)

    def test_prever_classifica_quando_confiante(self):
        _, texto = self.rodar("prever", "esqueci a senha da minha conta", "--limiar", "0.3")
        self.assertIn("classificar como 'acesso'", texto)

    def test_confusao_e_curva(self):
        _, texto = self.rodar("confusao")
        self.assertIn("classe real", texto)
        _, texto = self.rodar("curva")
        self.assertIn("cobertura", texto)

    def test_modelo_desconhecido(self):
        with self.assertRaises(SystemExit):
            self.rodar("confusao", "--modelo", "xyz")


if __name__ == "__main__":
    unittest.main()
