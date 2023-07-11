# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from lib2to3.pgen2 import token
import math
from operator import index
from turtle import forward
from typing import Dict, List, Optional

import torch
from torch import LongTensor, device
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

test_input = [
    "Time is recorded and water quantity measured in a v@@ at at the end .",
    "A better deal for British business is possible , and increasingly necessary as the eurozone embark@@ s on the road to closer economic and fiscal union .",
    "He admitted co@@ ca@@ ine possession at an earlier hearing .",
    "The match was of course a indication of what we are capable of .",
    "100 years ago , Heinrich Hans@@ j@@ ak@@ ob moved into his retirement home in Has@@ la@@ ch , on Sunday his final years in the &quot; Frei@@ hof &quot; were brought to life .",
    "The team &apos;s eye for the goal must be considerably improved .",
    "&quot; While the weaker aircraft deliveries were mostly anticipated , we are clearly disappointed by the margin performance in transportation , &quot; Do@@ erk@@ sen said in a client note .",
    "The E@@ cum@@ en@@ ical Dis@@ c@@ ussion Forum is an initiative of the Catholic and Protest@@ ant churches in Sal@@ em , the Sal@@ em Castle and the Lake Con@@ stance District Cultural Office .",
    "One trav@@ eller told the broad@@ c@@ aster CN@@ N about how many people had sought protection in pan@@ ic .",
    "However , in an interview , Blo@@ om has said that he and Ker@@ r still love each other .",
    "Built by experts",
    "I@@ ll@@ men@@ see : Second mountain bike race enth@@ uses participants",
    "Fron@@ tier &apos;s move to charge the car@@ r@@ y-@@ on fee if passengers don &apos;t buy direct from the airline is its latest effort to ste@@ er customers toward its own website .",
    "The secret of the ham is the mix of her@@ bs combined with salt .",
    "So@@ ya , o@@ at , al@@ mond or rice milk can be used in place of milk .",
    "It cannot account for assass@@ in@@ ating An@@ war al-@@ A@@ w@@ l@@ aki , an American citizen , without a trial , nor shi@@ r@@ king public funding and spending limits during presidential campaigns .",
    "Republic@@ an ob@@ struc@@ tion@@ ism cannot explain allowing the bu@@ gging of foreign leaders , nor having d@@ ron@@ es strike innocent children overseas .",
    "It will probably not be long before we understand why their tails sometimes go one way , sometimes the other",
    "D@@ itt@@ a , 42 , fed information to criminals because of fears his co@@ ca@@ ine ad@@ diction would be exposed",
    "That is what troub@@ les me most of all .",
    "The projects are planned to start in 20@@ 28 and 20@@ 34 .",
    "Spe@@ aking to an open government conference in London via video link , Mr K@@ erry said : &quot; There is no question that the President and I and others in government have actually learned of some things that had been happening on an automatic pilot because the ability has been there , going back to World War Two and to the very difficult years of the Cold War , and then , of course , 9 / 11 . &quot;",
    "N@@ ick Kro@@ ll rose to promin@@ ence on a deep@@ -@@ cable sit@@ com ( F@@ X@@ X &apos;s g@@ le@@ efully ra@@ un@@ chy fantas@@ y-@@ foot@@ b@@ all-@@ them@@ ed &quot; The League &quot; ) and now has his own Com@@ edy Central sk@@ etch show .",
    "It is perfect , but it lies .",
    "The fighting under@@ sc@@ ored the dangers the chemical weapons &apos; inspec@@ tors face as they race against tight deadlines in their mission to rid Syria of the toxic ar@@ senal in the midst of an ongoing civil war .",
    "Following a short flight from New York to Boston , the pupils have now arrived with their host families and are experiencing the school day at W@@ ey@@ mouth High School .",
    "At least six authorities globally - the European Commission , Fin@@ ma , Switzerland &apos;s competition authority W@@ ek@@ o , the F@@ CA , the Department of Justice in the US and the Hong Kong Monetary Authority - are looking at alleg@@ ations that ban@@ kers coll@@ u@@ ded to move the currencies market .",
    "Co-@@ operation with other lik@@ e-@@ minded countries will be easier in a non-@@ federal Europe of the Nations .",
    "USA : Sh@@ ots fi@@ red at Los Angeles Airport",
    "Finally , and most t@@ ell@@ ingly of all , our pol@@ l of business leaders found a clear majority wanted to see Britain pursue a course of treaty change and a relationship with the EU that is based on trade , not politics .",
    "Only when the psychological stra@@ in becomes severe do people give it consideration .",
    "What is certain is that the 19@@ -@@ year-old female driver of a V@@ W Golf was driving in the direction of Rev@@ ens@@ dorf and the 3@@ 8-@@ year-old man from G@@ ett@@ or@@ f came towards her in his Hy@@ un@@ da@@ i .",
    "Gir@@ ls who b@@ los@@ som early need re@@ assurance that , even when it happens ahead of schedule , the process is a normal part of life .",
    "&quot; It is not a matter of something we might choose to do , &quot; said Has@@ an I@@ k@@ hr@@ ata , executive director of the Southern California Ass@@ n@@ . of Govern@@ ments , which is planning for the state to start tracking miles driven by every California motor@@ ist by 20@@ 25 .",
    "&quot; There are several questions that must be answered or confirmed , &quot; he said .",
    "&quot; Property investments offer an attractive rate of return , &quot; said Ul@@ bri@@ ch .",
    "They earn as much as teachers and can make a good living .",
    "Among others , the airports in Kopenhagen , Frankfurt and Dubai have already done away with the noisy call@@ -@@ outs .",
    "Pat@@ ek may yet appeal his sentence .",
    "Here we find the residence of what was once the most powerful man to the north of the Alps , the Pra@@ et@@ or of the C@@ CA@@ A .",
    "This shows the &quot; desper@@ ation &quot; of the drug gangs , whose traditional routes have now been cut off , said Bill Sher@@ man of the D@@ EA Dru@@ g S@@ qu@@ ad in San Di@@ ego .",
    "In all the documentation le@@ aked by Mr Snow@@ den , there has , however , been no evidence to date that the US has passed on foreign companies &apos; trade sec@@ rets to its own companies .",
    "Happ@@ iness is an oasis that only dre@@ aming cam@@ els manage to reach",
    "The experience of Ther@@ esa R@@ ütten , head of the citizen &apos;s Life in Old Age service in the State Capital , is also that &apos; the majority of old people in Stuttgart want to remain in their own homes for as long as possible &apos; .",
    "&quot; According to current measurements , around 12@@ ,000 vehicles travel through the town of Gut@@ ach on the B@@ 33 on a daily basis , of which heavy goods traffic accounts for around ten per cent , &quot; emphasised Arn@@ old .",
    "&quot; Our current policy is that electronic devices cannot be used during tak@@ e-@@ off and landing and we have no immediate plans to change that , &quot; it said .",
    "To keep him alive , well , there &apos;s no reason to keep him alive .",
    "Staff gave evidence they observed bre@@ aches including Lord being alone with children , bab@@ ysi@@ t@@ ting them priv@@ ately , having them sit on his l@@ ap , saying he loved one and letting them play with his mobile phone .",
    "When the year comes to an end in just eight weeks , the anniversary year will be upon us .",
    "And I think about my father .",
    "That economic sp@@ ying takes place is not a surprise .",
    "The body was found the following night in the village of C@@ err@@ o Pel@@ ado , 15 kilometres to the north of the exclusive beach resort of Punta del E@@ ste .",
    "The other big issue is understandable : Par@@ ents simply don &apos;t want their very young dau@@ gh@@ ters having periods .",
    "Mayor R@@ alp@@ h Ger@@ ster than@@ ked Roth@@ mund and its team for the good work .",
    "Econom@@ ists are also worried about high levels of consumer debt in Thailand , which this week nar@@ row@@ ly emerged from technical recession .",
    "The S@@ N@@ P government in Scotland is - remark@@ ab@@ ly-@@ - the most successful anti-@@ austerity political movement in Europe , having won a spectacular majority in 2011 on the basis of opposing the cuts proposed ( and implemented ) by Labour &apos;s chan@@ cell@@ or Ali@@ sta@@ ir Dar@@ ling and the subsequent Tor@@ y-@@ Li@@ b Dem coalition .",
    "He says Obama has had worthy efforts th@@ war@@ ted by G@@ O@@ P ob@@ struc@@ tion@@ ism",
    "Without the possibility of domestic currency de@@ valu@@ ation , southern Europe finds itself with a built-in productivity disadvantage vis@@ -@@ à-@@ vis Germany .",
    "They want to see the Government priori@@ tise new trading links with the likes of China , India and Brazil , rather than getting bo@@ g@@ ged down in the long and ar@@ du@@ ous process of reform@@ ing the EU &apos;s ar@@ cane institutions .",
    "The two tra@@ ders would be the first R@@ BS employees to be suspended in the wi@@ dening probe that ech@@ oes the Lib@@ or inter@@ bank lending manipulation scandal .",
    "He said it was intended to avoid contamination of evidence but was &quot; over@@ ze@@ al@@ ous &quot; and poor@@ ly executed .",
    "The &quot; Drei@@ kö@@ nig &quot; sing@@ ers then made their own appearance , leading Hans@@ j@@ ak@@ ob to exc@@ la@@ im : &quot; O@@ h how beautiful , this brings childhood memories of my own time in a Drei@@ kö@@ nig group back to life &quot; .",
    "The Nas@@ da@@ q systems could not cope with the flood of purchase and sales orders , the Exchange Supervisory Authority established at a later date , imposing a record penalty of 10 million dollars on the company .",
    "However , there is concern that small and medium-sized companies remain vulnerable to h@@ acking and surveillance .",
    "As airport spo@@ kes@@ person Peter Kle@@ em@@ an announced to Radio Vienna that in adju@@ sting the approach towards announc@@ ements , Vienna Airport is following an international trend .",
    "Airbus says its ri@@ val is sti@@ cking to a seat concept from the 1950@@ s , when the average gir@@ th of the newly ch@@ ri@@ sten@@ ed &quot; jet set &quot; was nar@@ ro@@ wer .",
    "Managing director Gra@@ ham Tur@@ ner said F@@ light Centre had made 8 per cent profit in the first half and had started the second half strongly especially in Australian and UK non-@@ business travel .",
    "Exec@@ u@@ tives said new guidance would be provided in the fourth quarter .",
    "As for men , I expect we will be seeing a lot of z@@ om@@ bi@@ es , thanks to The Walking De@@ ad and I &apos;ll bet the D@@ af@@ t Pun@@ k space men will make it into our Inst@@ ag@@ ram feeds this year .",
    "&quot; We would welcome a review by C@@ ASA into allowing the use of electronic devices because we really do think it will improve the customer experience now that we have ( wireless in-@@ flight entertainment ) on our planes , &quot; a spo@@ kes@@ man said .",
    "In August , New York City agreed to end the practice of stor@@ ing the names and addresses of people whose cases are dis@@ missed after a police stop .",
    "Due to the low level of sunshine , from autumn until spring the village of R@@ ju@@ kan in the V@@ est@@ f@@ j@@ ord valley normally langu@@ ishes in the shad@@ ows of the surrounding mountains .",
    "He has not suffered any damage to his health or eating dis@@ orders - another wide-@@ spread assumption .",
    "As always , the south stand was left until the end , and finally the stand resumed its usual role as the lou@@ dest section of the stadium .",
    "In the case in question , a driver ignored a light that was permanently red after around three minutes and and was held responsible for neg@@ lig@@ ently j@@ umping a red light .",
    "Ha@@ ig@@ er@@ lo@@ ch : Focus on the Abend@@ mah@@ l@@ sk@@ ir@@ che",
    "Fron@@ tier said it will charge $ 25 if the fee is paid in advance , $ 100 if travelers wait to pay until they &apos;re at the gate .",
    "Conver@@ ting has had a positive effect for me .",
    "The incident took place in Terminal 3 . E@@ ye witn@@ esses reported seeing a sho@@ oter with a gun in one of the departure lo@@ ung@@ es , as was reported by several media .",
    "R@@ BS susp@@ ends two for@@ ex tra@@ ders",
    "The access roads were blocked off , which , according to CN@@ N , caused long tail@@ backs .",
    "They are said to have been active within the Ro@@ cker scene and to have at the time belong@@ ed to the Ban@@ di@@ dos .",
    "The waiting time must , however , be &quot; appropriate &quot; , which can be interpreted differently on a cas@@ e-@@ to-@@ case basis .",
    "Council sets its sights on rail system",
    "Due to the low level of sunshine , from autumn until spring the village of R@@ ju@@ kan in the V@@ est@@ f@@ j@@ ord valley normally langu@@ ishes in the shad@@ ows of the surrounding mountains .",
    "Bom@@ bar@@ di@@ er , the world &apos;s largest train@@ maker , said revenue in that division rose nearly 11 percent to $ 2.1 billion .",
    "&quot; In addition , we will have to wait and see whether rein@@ for@@ c@@ ements from the first team will be there , &quot; explained the S@@ G coach .",
    "Cross@@ le@@ y-@@ Holland , who has translated Be@@ ow@@ ul@@ f from Ang@@ lo@@ -S@@ ax@@ on as well as writing the Pen@@ gu@@ in Book of Nor@@ se My@@ ths and British Fol@@ k T@@ ales , said : &quot; You may well have intentions but you do better to keep them well out of sight . &quot;",
    "Pro@@ pos@@ als that are in the running ...",
    "His ad@@ diction to co@@ ca@@ ine left him hop@@ eless@@ ly com@@ promised and vulnerable to the mo@@ tives of leading members of organised crime groups who t@@ asked him to obtain valuable information regarding police investigations .",
    "Mr H@@ are said he had provided his view to the Y@@ M@@ CA N@@ S@@ W board that the lesson for the organisation from the &quot; Jon@@ athan Lord incident &quot; was &quot; not about reporting &quot; by staff , and the board agreed with him .",
    "&quot; After many years , we have now reached an agreement with the International Nuclear Energy Auth@@ orities to clear up the differences from past years , &quot; wrote Foreign Minister Moh@@ amm@@ ed D@@ sch@@ aw@@ ad S@@ ari@@ f on his Facebook page .",
    "&quot; This slow pace of flight testing - although in line with Bom@@ bar@@ di@@ er &apos;s internal schedule apparently - rein@@ forces our view that entr@@ y-@@ in@@ to-@@ service will be pushed to Q@@ 1 / 15 , &quot; said Do@@ erk@@ sen .",
    "N@@ ick Kro@@ ll rose to promin@@ ence on a deep@@ -@@ cable sit@@ com ( F@@ X@@ X &apos;s g@@ le@@ efully ra@@ un@@ chy fantas@@ y-@@ foot@@ b@@ all-@@ them@@ ed &quot; The League &quot; ) and now has his own Com@@ edy Central sk@@ etch show .",
    "Conver@@ ting has had a positive effect for me .",
    "He says Obama has had worthy efforts th@@ war@@ ted by G@@ O@@ P ob@@ struc@@ tion@@ ism",
    "&quot; Ei@@ gh@@ teen in@@ ches in seat width would be great for passengers , but the reality is that from a business point of the Airbus proposal is driven by the threat of the 7@@ 77 , &quot; said cabin inter@@ iors expert Mary Kir@@ by , founder and editor of the Run@@ way Gir@@ l Network .",
    "An overview of milk and egg alternatives",
    "Exper@@ ts say violence that left 14 adults and seven children dead is nothing more than random chance , not a sign of growing violence in America .",
    "All those involved will be happy with that evaluation .",
]


class PositionClassifier(nn.Module):
    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 256, out_dim: int = 40
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

        self.bias_ratio = []
        self.ratio_count = []
        for i in range(200):
            self.bias_ratio.append(torch.zeros((1,)).cuda())
            self.ratio_count.append(0)
        self.count = 0

        # cast model to double
        self.model.double()

        self.drop_probs = []

        self.analysis_info_list = []
        self.data_rec = []
        logger.info("begin inference")

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):

        # self.analysis_info_list.append([])
        analysis_info_list = []

        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        self.hack_input(net_input)

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        net_input["return_compositions"] = True
        encoder_outs = self.model.forward_encoder(net_input)

        encoder_compositions_states = None
        encoder_attn_states = None
        if "compositions_states" in encoder_outs[0]._fields:
            encoder_compositions_states = encoder_outs[0].compositions_states
        if "attn_states" in encoder_outs[0]._fields:
            encoder_attn_states = encoder_outs[0].attn_states
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # self.train_center_position_classifier(encoder_compositions_states, layer=5)
        # finalized = [[]]
        # finalized[0].append(
        #     {
        #         "tokens": torch.LongTensor([self.tgt_dict.eos()]).to(
        #             net_input["src_tokens"]
        #         ),
        #         "score": torch.zeros((1,)).to(device=net_input["src_tokens"].device),
        #         "attention": None,  # src_len x tgt_len
        #         "alignment": torch.empty(0),
        #         "positional_scores": None,
        #         "decomposition_positional_scores": None,
        #     }
        # )
        # return finalized

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        decomposition_scores = torch.zeros(
            bsz * beam_size, max_len + 1, 2 * max_len + 1
        )

        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            (
                lprobs,
                avg_attn_scores,
                compositions_logits,
                analysis_info,
            ) = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
                return_compositions=True,
            )
            analysis_info_list.append(analysis_info)

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            decomposition_scores = decomposition_scores.type_as(compositions_logits)

            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            beam_cand_indices = cand_indices + self.vocab_size * cand_beams

            decomposition_cand_scores = compositions_logits.view(
                bsz, -1, src_len + step + 2
            ).gather(
                dim=1,
                index=beam_cand_indices.unsqueeze(-1).expand(
                    -1, -1, src_len + step + 2
                ),
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                decomposition_eos_scores = decomposition_cand_scores[:, :beam_size][
                    eos_mask[:, :beam_size]
                ]

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                    decomposition_eos_scores,
                    decomposition_scores,
                    src_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                decomposition_cand_scores = decomposition_cand_scores[batch_idxs]

                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                decomposition_scores = decomposition_scores.view(
                    bsz, -1, decomposition_scores.size(-1)
                )[batch_idxs].view(
                    new_bsz * beam_size, -1, decomposition_scores.size(-1)
                )

                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                decomposition_scores[:, :step] = torch.index_select(
                    decomposition_scores[:, :step], dim=0, index=active_bbsz_idx
                )

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            decomposition_scores[:, :, : src_len + step + 2].view(
                bsz, beam_size, -1, src_len + step + 2
            )[:, :, step] = torch.gather(
                decomposition_cand_scores,
                dim=1,
                index=active_hypos.unsqueeze(-1).expand(-1, -1, src_len + step + 2),
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
            # pad_num = (src_tokens[sent] == self.tgt_dict.pad()).sum()
            # self.decomposition_positional_scores.append(
            #     finalized[sent][0]["decomposition_positional_scores"][:, pad_num:].cpu()
            # )
            # self.positional_scores.append(finalized[sent][0]["positional_scores"].cpu())

            # input_sum = finalized[sent][0]["decomposition_positional_scores"][
            #     :, 1 + pad_num :
            # ].sum(dim=-1)
            # bias_sum = finalized[sent][0]["decomposition_positional_scores"][:, 0]
            # for i in range(len(decomposition_positional_scores)):
            #     class_len = (decomposition_positional_scores.size(1) - 1 - pad_num) - (
            #         decomposition_positional_scores[i, pad_num + 1 :] == 0.0
            #     ).to(torch.long).sum().item()
            #     self.bias_ratio[class_len] += (
            #         bias_sum[i].abs() / (bias_sum[i] + input_sum[i]).abs()
            #     )
            #     self.ratio_count[class_len] += 1

        decomposition_positional_scores = finalized[0][0][
            "decomposition_positional_scores"
        ]
        net_input = sample["net_input"]

        # self.compute_scores(
        #     decomposition_positional_scores,
        #     net_input,
        #     finalized[0][0]["tokens"],
        #     finalized[0][0]["positional_scores"],
        #     bos_token,
        #     drop_rate=0.1,
        # )

        self.count += len(sample["id"])
        # import pdb

        # pdb.set_trace()
        sample_num = len(self.src_strs)

        self.decoder_norm_logit_analysis(
            analysis_info_list,
            finalized[0][0]["tokens"],
            sample_num=sample_num,
            file_name="position_decoder_norms_logits.pt",
        )

        # self.decoder_attn_analysis(
        #     analysis_info_list,
        #     sample_num=sample_num,
        #     file_name="nh_test_decoder_attn.pt",
        # )
        # self.decoder_norm_attribution(
        #     analysis_info_list,
        #     net_input,
        #     finalized[0][0]["tokens"],
        #     finalized[0][0]["positional_scores"],
        #     bos_token,
        #     drop_rate=0.1,
        # )

        if encoder_compositions_states is not None:
            self.encoder_norm_analysis(
                encoder_compositions_states,
                sample_num=sample_num,
                file_name="position_encoder_norms.pt",
            )
        # if encoder_attn_states is not None:
        #     self.encoder_attn_analysis(
        #         encoder_attn_states,
        #         sample_num=sample_num,
        #         file_name="nh_test_encoder_attn.pt",
        #     )
        # self.decoder_norm_analysis(
        #     analysis_info_list, sample_num=1, file_name="hallucination_decoder_norms.pt"
        # )

        # self.tail_peak_record(
        #     src_tokens[0],
        #     finalized[0][0]["tokens"],
        #     decomposition_positional_scores,
        #     analysis_info_list,
        #     sample_num=3003,
        # )

        return finalized

    def train_center_position_classifier(
        self, encoder_compositions_states: List[Tensor], layer: int, max_len: int = 40
    ):
        def init_model(input_dim, hidden_dim, out_dim, device):
            if "position_classifier" not in dir(self):
                self.position_classifier = PositionClassifier(
                    input_dim, hidden_dim, out_dim
                ).to(device=device)
                self.optm = torch.optim.Adam(
                    self.position_classifier.parameters(), lr=0.01
                )
                self.scheduler = sc = torch.optim.lr_scheduler.LinearLR(
                    self.optm, 1.0, 0.2, 3000
                )
                self.train_num = 0

        def test_model(device):
            precision_num = 0
            all_num = 0
            for index in range(100):
                net_input = {}
                net_input["return_compositions"] = True
                net_input["src_tokens"] = torch.LongTensor([[1, 2]]).to(device=device)
                net_input["src_lengths"] = torch.LongTensor([[2]]).to(device=device)
                self.hack_input(net_input, index)
                encoder_outs = self.model.forward_encoder(net_input)
                encoder_compositions_states = encoder_outs[0].compositions_states
                center_position_states, label = data_preprocess(
                    encoder_compositions_states=encoder_compositions_states
                )
                with torch.no_grad():
                    self.position_classifier.eval()
                    y: Tensor = self.position_classifier(center_position_states.float())
                    precision_num += (label == y.max(dim=-1)[1]).sum().item()
                    all_num += len(label)
            precision = precision_num / all_num
            logger.info(f"test precision: {100*precision} %")

        def data_preprocess(
            encoder_compositions_states: List[Tensor], use_decompositon=True
        ):
            select_states = encoder_compositions_states[layer]
            encoder_len, _, _, hidden_dim = select_states.size()

            init_model(hidden_dim, 256, max_len, select_states.device)

            select_states = select_states.view(
                encoder_len, encoder_len + 1, 2, hidden_dim
            )
            select_len = min(encoder_len, max_len)
            center_index = list(range(select_len))

            if use_decompositon:
                # select_states = select_states[:select_len, 1 : select_len + 1, 1]
                select_states = select_states[:select_len, 1 : select_len + 1].sum(
                    dim=2
                )
                center_position_states = select_states[center_index, center_index]
            else:
                center_position_states = select_states[:select_len].sum(dim=(1, 2))

            label = torch.LongTensor(center_index).to(device=select_states.device)
            return center_position_states, label

        center_position_states, label = data_preprocess(
            encoder_compositions_states=encoder_compositions_states
        )
        with torch.set_grad_enabled(True):
            self.position_classifier.train(True)
            self.optm.zero_grad()
            y: Tensor = self.position_classifier(center_position_states.float())
            loss = F.cross_entropy(y, label)
            loss.backward()
            self.optm.step()
            self.scheduler.step()
            self.train_num += 1

        if self.train_num % 200 == 0:
            logger.info(f"train process: {self.train_num}/3000")
            test_model(device=encoder_compositions_states[0].device)

    def tail_peak_record(
        self,
        src_tokens,
        tokens,
        decomposition_positional_scores,
        analysis_info_list,
        sample_num=10,
        file_name="tail_peak.pt",
    ):
        if "logit_list" not in dir(self):
            self.logit_list = []
        if "norm_list" not in dir(self):
            self.norm_list = []

        prev_tokens = torch.zeros_like(tokens).to(tokens)
        prev_tokens[0] = self.tgt_dict.bos()
        prev_tokens[1:] = tokens[:-1]

        def remove_bpe_word(tokens, pos):
            def get_word_str(word_id):
                if word_id == self.tgt_dict.eos():
                    return "<eos>"
                elif word_id == self.tgt_dict.bos():
                    return "<bos>"
                elif word_id == self.tgt_dict.unk():
                    return "<unk>"
                elif word_id == self.tgt_dict.pad():
                    return "<pad>"
                else:
                    return self.tgt_dict.string([word_id])

            res_list = []
            # forward search
            for i in range(pos - 1, -1, -1):
                word_str = get_word_str(tokens[i])
                if word_str.endswith("@@"):
                    res_list.insert(0, word_str[:-2])
                else:
                    break
            # backward search
            for i in range(pos, len(tokens)):
                word_str = get_word_str(tokens[i])
                if word_str.endswith("@@"):
                    res_list.append(word_str[:-2])
                else:
                    res_list.append(word_str)
                    break
            return "".join(res_list)

        def extract_data(scores_mat: Tensor):
            pos_range = min(5, scores_mat.size(0))
            topk = 3
            src_len = scores_mat.size(1) - 1 - scores_mat.size(0)

            scores_mat: Tensor = scores_mat[-pos_range:, 1:]
            _, top_index = scores_mat.topk(topk, dim=-1)

            score_pos_list = []

            for i in range(pos_range):
                score_dict = {}
                pos_top_index = top_index[i]
                score_dict["is_tgt"] = pos_top_index >= src_len
                score_dict["top_word"] = []
                for j in range(len(pos_top_index)):
                    if score_dict["is_tgt"][j]:
                        score_dict["top_word"].append(
                            remove_bpe_word(prev_tokens, pos_top_index[j] - src_len)
                        )
                    else:
                        score_dict["top_word"].append(
                            remove_bpe_word(src_tokens, pos_top_index[j])
                        )
                score_pos_list.append(score_dict)
            return score_pos_list

        logit_pos_list = extract_data(decomposition_positional_scores)
        self.logit_list.append(logit_pos_list)

        compositions_states_norms = torch.zeros(
            (
                len(analysis_info_list),
                analysis_info_list[-1]["compositions_states"][0].size(1),
            )
        )
        for j, analysis_info in enumerate(analysis_info_list):
            compositions = analysis_info["compositions_states"][-1]
            compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                compositions, dim=-1
            )[0, :, 0]

        norm_pos_list = extract_data(compositions_states_norms)
        self.norm_list.append(norm_pos_list)

        if self.count == sample_num:
            torch.save((self.logit_list, self.norm_list), file_name)

    def hack_input(self, net_input, index=None):
        def load_nh_test_data():
            if "src_strs" not in dir(self):
                self.src_strs = []
            else:
                return
            base_path = "/home/yangs/code/fairseq_models/attribution_transformer_wmt16/hallucination/"
            with open(base_path + "test_normal.txt", "r") as normal_file:
                for line in normal_file:
                    self.src_strs.append(line.strip())
            with open(base_path + "test_nh.txt", "r") as nh_file:
                for line in nh_file:
                    self.src_strs.append(line.strip())
            assert len(self.src_strs) == 100

        self.src_strs = [
            "I hope he sees what I am doing .",
            "A lot of my clients are very young .",
            "The price of this ship is very nice .",
            "The room of this ship is very nice .",
            "The room of this hotel is very nice .",
        ]

        # load_nh_test_data()
        if index is None:
            index = self.count

        if index < len(self.src_strs):
            src_string = self.src_strs[index]
        else:
            exit()
        new_src_tokens = self.tgt_dict.encode_line(
            src_string, add_if_not_exist=False
        ).to(net_input["src_tokens"])
        assert not torch.any(new_src_tokens == self.tgt_dict.unk())
        new_src_tokens = new_src_tokens[None]
        net_input["src_tokens"] = new_src_tokens
        net_input["src_lengths"][0] = new_src_tokens.size(-1)

    def hack_input_test_classifier(self, net_input, index=None):
        def load_nh_test_data():
            if "src_strs" not in dir(self):
                self.src_strs = []
            else:
                return
            base_path = "/home/yangs/code/fairseq_models/attribution_transformer_wmt16/hallucination/"
            with open(base_path + "test_normal.txt", "r") as normal_file:
                for line in normal_file:
                    self.src_strs.append(line.strip())
            with open(base_path + "test_nh.txt", "r") as nh_file:
                for line in nh_file:
                    self.src_strs.append(line.strip())
            assert len(self.src_strs) == 100

        # src_strs = [
        #     "I hope he sees what I am doing .",
        #     "A lot of my clients are very young .",
        #     "The price of this ship is very nice .",
        #     "The room of this ship is very nice .",
        #     "The room of this hotel is very nice .",
        # ]

        # load_nh_test_data()
        if index is None:
            index = self.count

        if index < len(test_input):
            src_string = test_input[index]
        else:
            exit()
        new_src_tokens = self.tgt_dict.encode_line(
            src_string, add_if_not_exist=False
        ).to(net_input["src_tokens"])
        assert not torch.any(new_src_tokens == self.tgt_dict.unk())
        new_src_tokens = new_src_tokens[None]
        net_input["src_tokens"] = new_src_tokens
        net_input["src_lengths"][0] = new_src_tokens.size(-1)

    def decoder_attn_analysis(
        self, analysis_info_list, sample_num=10, file_name="cross_attn_data_rec.pt"
    ):
        if "cross_attn_list" not in dir(self):
            self.cross_attn_list = []
        cross_attn_layers = []
        for i in range(len(analysis_info_list[0]["attn_states"])):
            cross_attn_states = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[i]["attn_states"][0].size(-1),
                )
            ).cpu()
            for j, analysis_info in enumerate(analysis_info_list):
                cross_attn = analysis_info["attn_states"][i]
                cross_attn_states[j] = cross_attn[0, 0].cpu()
            cross_attn_layers.append(cross_attn_states)
        self.cross_attn_list.append(cross_attn_layers)
        if self.count == sample_num:
            torch.save(self.cross_attn_list, file_name)

    def decoder_norm_attribution(
        self,
        analysis_info_list: List[Dict],
        net_input,
        pred_tokens: LongTensor,
        positional_scores: Tensor,
        bos_token: Optional[int] = None,
        drop_rate=0.1,
    ):
        compositions_states_norms = torch.zeros(
            (
                len(analysis_info_list),
                analysis_info_list[-1]["compositions_states"][0].size(1),
            )
        )
        for j, analysis_info in enumerate(analysis_info_list):
            compositions = analysis_info["compositions_states"][-1]
            compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                compositions, dim=-1
            )[0, :, 0]

        self.compute_scores(
            compositions_states_norms,
            net_input,
            pred_tokens,
            positional_scores,
            bos_token,
            drop_rate,
        )
        self.AOPC()

    def decoder_norm_logit_analysis(
        self,
        analysis_info_list,
        pred_tokens,
        sample_num=10,
        file_name="decoder_norms_logits_data_rec.pt",
    ):
        if "decoder_compositions_states_norms_list" not in dir(self):
            self.decoder_compositions_states_norms_list = []
        if "decoder_compositions_states_logit_list" not in dir(self):
            self.decoder_compositions_states_logit_list = []
        compositions_states_norms_layers = []
        compositions_states_logits_layers = []
        for i in range(len(analysis_info_list[0]["compositions_states"])):
            compositions_states_norms = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[-1]["compositions_states"][0].size(1),
                )
            ).cpu()
            compositions_states_logits = torch.zeros_like(
                compositions_states_norms
            ).cpu()
            assert len(pred_tokens) == len(analysis_info_list)
            for j, analysis_info in enumerate(analysis_info_list):
                compositions = analysis_info["compositions_states"][i]
                compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                    compositions, dim=-1
                ).cpu()[0, :, 0]
                token_embed = self.model.models[0].decoder.embed_tokens.weight[
                    pred_tokens[j]
                ]
                compositions_states_logits[j, : compositions.size(1)] = torch.matmul(
                    compositions[0, :, 0], token_embed
                ).cpu()
            compositions_states_norms_layers.append(compositions_states_norms)
            compositions_states_logits_layers.append(compositions_states_logits)
        self.decoder_compositions_states_norms_list.append(
            compositions_states_norms_layers
        )
        self.decoder_compositions_states_logit_list.append(
            compositions_states_logits_layers
        )
        if self.count == sample_num:
            torch.save(
                (
                    self.decoder_compositions_states_norms_list,
                    self.decoder_compositions_states_logit_list,
                ),
                file_name,
            )

    def decoder_norm_analysis(
        self, analysis_info_list, sample_num=10, file_name="decoder_norms_data_rec.pt"
    ):
        if "decoder_compositions_states_norms_list" not in dir(self):
            self.decoder_compositions_states_norms_list = []
        compositions_states_norms_layers = []
        for i in range(len(analysis_info_list[0]["compositions_states"])):
            compositions_states_norms = torch.zeros(
                (
                    len(analysis_info_list),
                    analysis_info_list[-1]["compositions_states"][0].size(1),
                )
            ).cpu()
            for j, analysis_info in enumerate(analysis_info_list):
                compositions = analysis_info["compositions_states"][i]
                compositions_states_norms[j, : compositions.size(1)] = torch.norm(
                    compositions, dim=-1
                ).cpu()[0, :, 0]
            compositions_states_norms_layers.append(compositions_states_norms)
        self.decoder_compositions_states_norms_list.append(
            compositions_states_norms_layers
        )
        if self.count == sample_num:
            torch.save(self.decoder_compositions_states_norms_list, file_name)

    def encoder_attn_analysis(
        self, attn_states, sample_num=10, file_name="encoder_attn_data_rec.pt"
    ):
        if "encoder_attn_list" not in dir(self):
            self.encoder_attn_list = []
        encoder_attn_states = []
        for layer_attn in attn_states:
            encoder_attn_states.append(layer_attn[0].cpu())
        self.encoder_attn_list.append(encoder_attn_states)
        if self.count == sample_num:
            torch.save(self.encoder_attn_list, file_name)

    def encoder_norm_analysis(
        self, compositions_states, sample_num=10, file_name="encoder_norms_data_rec.pt"
    ):
        compositions_states_norms = []
        for compositions in compositions_states:
            compositions_states_norms.append(
                torch.norm(compositions, dim=-1).squeeze(-1).cpu()
            )
        if "encoder_compositions_states_norms_list" not in dir(self):
            self.encoder_compositions_states_norms_list = []
        self.encoder_compositions_states_norms_list.append(compositions_states_norms)
        if self.count == sample_num:
            torch.save(self.encoder_compositions_states_norms_list, file_name)

    def AOPC(self):
        if self.count % 100 == 0:
            temp_drop_probs = torch.stack(self.drop_probs)
            logger.info("AOPC: {}".format(torch.mean(temp_drop_probs).item()))

        # all_positional_scores = [self.positional_scores, self.isolate_positional_scores]
        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:
            logger.info("end inference")
            drop_probs = torch.stack(self.drop_probs)
            analysis_info_tensor = torch.stack(self.analysis_info_list)
            logger.info("avg coef: {}".format(torch.mean(analysis_info_tensor).item()))
            logger.info("AOPC: {}".format(torch.mean(drop_probs).item()))
            # for i in range(len(self.bias_ratio)):
            #     print((self.bias_ratio[i] / self.ratio_count[i]).item())
            # for i in range(len(self.bias_ratio)):
            #     print(self.ratio_count[i])
            # print("avg:", torch.cat(self.bias_ratio, dim=0).sum()/sum(self.ratio_count))

    def bos_difference(self, analysis_info_list):
        def check_head_word(compositions_logits):
            pred_index = torch.argmax(compositions_logits.sum(dim=2), dim=-1)[
                0, 0
            ].item()
            head_word = self.tgt_dict.string([pred_index])
            lower_head_word = head_word.lower()
            if lower_head_word == head_word:
                return None
            lower_word_index = self.tgt_dict.index(lower_head_word)
            if lower_word_index == self.tgt_dict.unk():
                return None
            return pred_index, lower_word_index

        compositions_logits = analysis_info_list[0]["compositions_logits"]
        res = check_head_word(compositions_logits)
        if res is not None:
            pred_index, lower_word_index = res
            wo_dif = (
                compositions_logits[0, 0, 1:-1, pred_index].sum()
                - compositions_logits[0, 0, 1:-1, lower_word_index].sum()
            )
            w_dif = (
                compositions_logits[0, 0, 1:, pred_index].sum()
                - compositions_logits[0, 0, 1:, lower_word_index].sum()
            )
            self.data_rec.append((wo_dif.item(), w_dif.item()))

        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:

            wo_dif_sum = 0
            w_dif_sum = 0
            for wo_dif, w_dif in self.data_rec:
                wo_dif_sum += wo_dif
                w_dif_sum += w_dif
            torch.save(self.data_rec, "data_rec.pt")
            print("avg w/o dif", wo_dif_sum / len(self.data_rec))
            print("avg w dif", w_dif_sum / len(self.data_rec))

    def norm_analysis(self):
        if self.count == 3003 or self.count == 6750:  # or self.count > 3000:
            for sample_analysis_info in self.analysis_info_list:
                for analysis_info in sample_analysis_info:
                    analysis_info["composition_norms"] = analysis_info[
                        "composition_norms"
                    ].cpu()
            torch.save(self.analysis_info_list, "analysis.pt")

    def compute_scores(
        self,
        decomposition_positional_scores: Tensor,
        net_input,
        pred_tokens: LongTensor,
        positional_scores: Tensor,
        bos_token: Optional[int] = None,
        drop_rate=0.1,
    ):
        decomposition_positional_scores = decomposition_positional_scores[:, 1:]
        positional_probs = torch.exp(positional_scores)

        bos_token = self.eos if bos_token is None else bos_token
        prefix_tokens = torch.zeros_like(pred_tokens).to(pred_tokens)
        prefix_tokens[0] = bos_token
        prefix_tokens[1:] = pred_tokens[:-1]

        pred_len = decomposition_positional_scores.size(0)
        src_tokens = net_input["src_tokens"]

        bsz, src_len = src_tokens.size()[:2]
        assert bsz == 1
        pad_num = (src_tokens[0] == self.tgt_dict.pad()).sum()
        assert pad_num == 0

        assert src_len + pred_len == decomposition_positional_scores.size(1)

        sample_drop_probs = []

        for i in range(pred_len):
            tokens_to_keep = max(int(round((src_len + i + 1) * drop_rate)), 1)
            _, top_indices = torch.topk(
                decomposition_positional_scores[i, : src_len + i + 1], k=tokens_to_keep
            )
            encoder_dropout_tokens = top_indices[top_indices < src_len].unsqueeze(dim=0)
            decoder_dropout_tokens = top_indices[~(top_indices < src_len)].unsqueeze(
                dim=0
            )
            if encoder_dropout_tokens.size(1) == 0:
                encoder_dropout_tokens = None
            if decoder_dropout_tokens.size(1) == 0:
                decoder_dropout_tokens = None
            else:
                decoder_dropout_tokens -= src_len
            assert (
                encoder_dropout_tokens is not None or decoder_dropout_tokens is not None
            )
            net_input["return_compositions"] = False
            net_input["dropout_tokens"] = encoder_dropout_tokens
            encoder_outs = self.model.forward_encoder(net_input)
            lprobs, avg_attn_scores, _, _ = self.model.forward_decoder(
                prefix_tokens[None, : i + 1],
                encoder_outs,
                incremental_states=[None],
                temperature=self.temperature,
                return_compositions=False,
                dropout_tokens=decoder_dropout_tokens,
            )

            sample_drop_probs.append(
                positional_probs[i] - torch.exp(lprobs[0, pred_tokens[i]])
            )

        self.drop_probs.append(torch.stack(sample_drop_probs).mean())

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
        decomposition_eos_scores,
        decomposition_scores,
        src_len,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]

        decomposition_pos_scores = decomposition_scores.index_select(0, bbsz_idx)[
            :, : step + 1, : src_len + step + 2
        ]

        pos_scores[:, step] = eos_scores

        decomposition_pos_scores[:, step] = decomposition_eos_scores

        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "decomposition_positional_scores": decomposition_pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
        self,
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        return_compositions: bool = False,
        dropout_tokens: Tensor = None,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                    return_compositions=return_compositions,
                    dropout_tokens=dropout_tokens,
                )
            else:
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    return_compositions=return_compositions,
                    dropout_tokens=dropout_tokens,
                )

            attn: Optional[Tensor] = None
            compositions_logits = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                compositions_logits = decoder_out[1]["compositions_logits"]
                analysis_info = decoder_out[1]["analysis_info"]
                if analysis_info is not None:
                    analysis_info["attn_states"] = decoder_out[1]["attn_states"]
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]

            if return_compositions:
                compositions_logits = compositions_logits[:, -1, :, :]
                compositions_logits = compositions_logits.transpose(
                    1, 2
                ).contiguous()  # bsz*beam x dict_size x src_len+gen_len

            if self.models_size == 1:
                return probs, attn, compositions_logits, analysis_info

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(SequenceGenerator):
    def __init__(self, models, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
